import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import default_data_collator
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from utils import *
import os
from datasets import concatenate_datasets
# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Core training function

def do_train(args, model, train_dataloader, save_dir="./out"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Optimizer
    lr = getattr(args, "learning_rate", 5e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    first_batch = next(iter(train_dataloader))
    if "token_type_ids" in first_batch:
        print("[WARN] token_type_ids present in dataloader batch -> dropping for safety")
        first_batch.pop("token_type_ids")

    for key in ("input_ids", "attention_mask", "labels"):
        if key not in first_batch:
            raise RuntimeError(f"Batch missing required key: {key}")
        if not torch.is_tensor(first_batch[key]):
            raise RuntimeError(f"Batch field {key} is not a torch.Tensor")

    B, L = first_batch["input_ids"].shape
    print(f"[sanity] batch shape: input_ids={tuple(first_batch['input_ids'].shape)}, "
          f"attn={tuple(first_batch['attention_mask'].shape)}, labels={tuple(first_batch['labels'].shape)}")
    print(f"[sanity] dtypes: ids={first_batch['input_ids'].dtype}, attn={first_batch['attention_mask'].dtype}, "
          f"labels={first_batch['labels'].dtype}")

    
    vocab_size = model.get_input_embeddings().weight.shape[0]
    ids_min = int(first_batch["input_ids"].min().item())
    ids_max = int(first_batch["input_ids"].max().item())
    print(f"[sanity] input_ids min/max: {ids_min}/{ids_max}, vocab_size={vocab_size}")
    if ids_min < 0 or ids_max >= vocab_size:
        raise RuntimeError(f"input_ids out of range: [{ids_min}, {ids_max}] vs vocab_size {vocab_size}")

    batch0 = {k: v.to(device, non_blocking=False) for k, v in first_batch.items()}
    with torch.cuda.device_of(next(model.parameters())) if device.type == "cuda" else nullcontext():  # type: ignore
        try:
            out0 = model(
                input_ids=batch0["input_ids"],
                attention_mask=batch0["attention_mask"],
                labels=batch0["labels"].long(),  # ensure CE expects Long
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(f"[sanity] first forward ok. loss={float(out0.loss.item()):.4f}")
        except Exception as e:
            print("[sanity] first forward FAILED; enabling CUDA sync and re-raising...")
            if device.type == "cuda":
                torch.cuda.synchronize()
            raise

    # ==== Training loop =====================================================
    num_epochs = getattr(args, "num_epochs", 3)
    step = 0
    for epoch in range(num_epochs):
        print(f"[train] epoch {epoch+1}/{num_epochs}")
        for batch in train_dataloader:
            batch.pop("token_type_ids", None)

            feed = {
                "input_ids": batch["input_ids"].to(device, non_blocking=False),
                "attention_mask": batch["attention_mask"].to(device, non_blocking=False),
                "labels": batch["labels"].long().to(device, non_blocking=False),
            }

            opt.zero_grad(set_to_none=True)
            try:
                outputs = model(**feed)
                loss = outputs.loss
                loss.backward()
                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
            except Exception:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                raise

            step += 1
            if step % 50 == 0:
                print(f"[train] step {step}  loss={float(loss.item()):.4f}")

    os.makedirs(save_dir, exist_ok=True)
    try:
        model.save_pretrained(save_dir)
    except Exception as e:
        print(f"[WARN] failed to save model: {e}")


    #print("Training completed...")
    #print("Saving Model....")
    #model.save_pretrained(save_dir)

    return

      


# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file):
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()

    metric = evaluate.load("accuracy")
    out_file = open(out_file, "w")

    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

        # write to output file
        for pred, label in zip(predictions, batch["labels"]):
                out_file.write(f"{pred.item()}\n")
                out_file.write(f"{label.item()}\n")
    out_file.close()
    score = metric.compute()

    return score


# Created a dataloader for the augmented training dataset
from transformers import AutoTokenizer, default_data_collator
from datasets import concatenate_datasets
import random, torch
from typing import Any, Dict

# --- adapt to your transform in utils.py ---
try:
    from utils import custom_transform as _user_custom_transform
except Exception as e:
    _user_custom_transform = None
    print(f"[WARN] Could not import utils.custom_transform: {e}")

def _apply_user_transform_to_text(text: str) -> str:
    """
    Accepts a raw string, calls utils.custom_transform(example)
    where example may be expected as a dict or string.
    Returns a transformed string.
    """
    if _user_custom_transform is None:
        # identity if not found
        return text

    # Try dict-style first: {"text": text}
    try:
        out = _user_custom_transform({"text": text})
        if isinstance(out, dict) and "text" in out:
            return out["text"]
        if isinstance(out, str):
            return out
    except Exception:
        pass

    # Try string-style: custom_transform(text)
    try:
        out = _user_custom_transform(text)
        if isinstance(out, dict) and "text" in out:
            return out["text"]
        if isinstance(out, str):
            return out
    except Exception as e:
        print(f"[WARN] custom_transform failed on '{text[:50]}...': {e}")

    # Fallback: identity
    return text


def create_augmented_dataloader(args, dataset):
    # --- tokenizer ---
    model_name = getattr(args, "model_name", None) or "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    max_len  = getattr(args, "max_length", 128)
    batch_sz = getattr(args, "batch_size", 8)
    g = torch.Generator()
    if getattr(args, "seed", None) is not None:
        random.seed(args.seed)
        g.manual_seed(args.seed)

    # --- base split/columns ---
    ds_train = dataset["train"]
    cols = set(ds_train.column_names)
    text_col = "text" if "text" in cols else "sentence" if "sentence" in cols else None
    if text_col is None:
        raise RuntimeError(f"No text column found. Columns: {ds_train.column_names}")
    label_col = "label" if "label" in cols else "labels" if "labels" in cols else None
    if label_col is None:
        raise RuntimeError(f"No label column found. Columns: {ds_train.column_names}")

    # --- pick 5k rows and transform their text with utils.custom_transform ---
    n_aug = 5000
    n = len(ds_train)
    idx = list(range(n))
    aug_idx = random.sample(idx, n_aug) if n >= n_aug else [random.choice(idx) for _ in range(n_aug)]
    ds_aug_src = ds_train.select(aug_idx)

    def map_transform(batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            text_col: [_apply_user_transform_to_text(t) for t in batch[text_col]],
            label_col: batch[label_col],
        }

    ds_aug_text = ds_aug_src.map(map_transform, batched=True, desc="Transform 5k aug samples")

    # --- tokenize; drop token_type_ids explicitly ---
    def tok_map(batch: Dict[str, Any]) -> Dict[str, Any]:
        enc = tokenizer(
            batch[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        # Hugging Face trainers expect 'labels'
        enc["labels"] = batch[label_col]
        # remove token_type_ids to avoid bad indices on BERT-style token_type_embeddings
        if "token_type_ids" in enc:
            enc.pop("token_type_ids")
        return enc

    tok_train = ds_train.map(
        tok_map,
        batched=True,
        remove_columns=ds_train.column_names,
        desc="Tokenize original train",
    )
    tok_aug = ds_aug_text.map(
        tok_map,
        batched=True,
        remove_columns=ds_aug_text.column_names,
        desc="Tokenize augmented train",
    )

    train_tok = concatenate_datasets([tok_train, tok_aug])
    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Windows-safe DataLoader: num_workers=0, pin_memory=False
    train_dataloader = torch.utils.data.DataLoader(
        train_tok,
        batch_size=batch_sz,
        shuffle=True,
        collate_fn=default_data_collator,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    print(f"[create_augmented_dataloader] final train size: {len(train_tok)} "
        f"(orig {len(tok_train)} + aug {len(tok_aug)})")
    return train_dataloader




# Create a dataloader for the transformed test set
def create_transformed_dataloader(args, dataset, debug_transformation):
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('=' * 30)

        exit()

    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset
    eval_dataloader = DataLoader(transformed_val_dataset, batch_size=args.batch_size)

    return eval_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true",
                        help="use a subset for training to debug your training loop")
    parser.add_argument("--debug_transformation", action="store_true",
                        help="print a few transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    global device
    global tokenizer

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Tokenize the dataset
    dataset = load_dataset("imdb")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    # Create dataloaders for iterating over the dataset
    if args.debug_train:
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size)
        print(f"Debug training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")
    else:
        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args.batch_size)
        print(f"Actual training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")

    # Train model on the original training dataset
    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out")
        # Change eval dir
        args.model_dir = "./out"

    # Train model on the augmented training dataset
    if args.train_augmented:
        train_dataloader = create_augmented_dataloader(args, dataset)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out_augmented")
        # Change eval dir
        args.model_dir = "./out_augmented"

    # Evaluate the trained model on the original test dataset
    if args.eval:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_original.txt"
        score = do_eval(eval_dataloader, args.model_dir, out_file)
        print("Score: ", score)

    # Evaluate the trained model on the transformed test dataset
    if args.eval_transformed:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_transformed.txt"
        eval_transformed_dataloader = create_transformed_dataloader(args, dataset, args.debug_transformation)
        score = do_eval(eval_transformed_dataloader, args.model_dir, out_file)
        print("Score: ", score)
