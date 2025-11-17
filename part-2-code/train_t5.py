import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#PAD_IDX = 0
TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small")
PAD_IDX = TOKENIZER.pad_token_id


def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'ft_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
       
        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()

        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # IMPORTANT: mask pad tokens for T5 loss
        labels = decoder_targets.clone()
        labels[labels == PAD_IDX] = -100   # T5 ignores -100 automatically

        # Let T5 compute the correct cross-entropy loss
        out = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=labels,    # decoder_inputs are handled automatically
        )

        loss = out.loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Accumulate statistics
        with torch.no_grad():
            # number of non-pad tokens
            num_tokens = torch.sum(decoder_targets != PAD_IDX).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens

        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    printed = 0 
    all_preds = []
    all_targets = []
    total_loss = 0
    total_tokens = 0
    syntax_errors = 0

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, init_dec_inp in dev_loader:

            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # ---------- Compute loss ----------
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            ).logits

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

            total_loss += loss.item() * non_pad.sum().item()
            total_tokens += non_pad.sum().item()

            # ---------- Generate SQL (greedy decoding) ----------
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=1,
                early_stopping=True,
            )

            # Decode predictions + targets into strings
        
            for pred_ids, tgt_ids in zip(generated, decoder_targets):
                pred_sql = TOKENIZER.decode(pred_ids, skip_special_tokens=True)
                tgt_sql = TOKENIZER.decode(tgt_ids, skip_special_tokens=True)

                all_preds.append(pred_sql)
                all_targets.append(tgt_sql)

                # if printed < 5:
                #     print("\n================ SAMPLE {} ================".format(printed + 1))
                #     print("Natural Language Input:")
                #     print(TOKENIZER.decode(encoder_input[0], skip_special_tokens=True))

                #     print("\nGOLD SQL:")
                #     print(tgt_sql)

                #     print("\nPREDICTED SQL:")
                #     print(pred_sql)
                #     print("=========================================\n")

                # printed += 1


                # Syntax error: try executing SQL
                try:
                    compile(pred_sql, "<string>", "exec")
                except Exception:
                    syntax_errors += 1
                


    # ---------- Save output SQL to file ----------
    with open(model_sql_path, "w", encoding="utf-8") as f:
        for q in all_preds:
            f.write(q + "\n")

    # ---------- Save + compute record-level metrics ----------
    # These helper functions are provided by the assignment.
    save_queries_and_records(all_preds, model_sql_path, model_record_path)
    record_f1, record_em, sql_em, error_info = compute_metrics(
        gt_sql_pth, model_sql_path,
        gt_record_path, model_record_path
    )
    if isinstance(error_info, list):
    # count how many queries had a non-empty error message
        num_errors = sum(1 for e in error_info if isinstance(e, str) and e.strip() != "")
        error_rate = num_errors / len(error_info) if len(error_info) > 0 else 0.0
    else:
        # fallback if compute_metrics ever changes to return a scalar
        try:
            error_rate = float(error_info)
        except Exception:
            print(f"Warning: could not interpret error_info={error_info}, defaulting error_rate=0.")
            error_rate = 0.0

    eval_loss = total_loss / total_tokens
    return eval_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    all_preds = []

    with torch.no_grad():
        for encoder_input, encoder_mask, init_dec_inp in test_loader:

            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=128,
                num_beams=1,
                early_stopping=True,
            )

            for pred_ids in generated:
                pred_sql = TOKENIZER.decode(pred_ids, skip_special_tokens=True)
                all_preds.append(pred_sql)

    # Save SQL predictions
    with open(model_sql_path, "w", encoding="utf-8") as f:
        for q in all_preds:
            f.write(q + "\n")

    # Save query + record results (assignment-provided helper)
    save_queries_and_records(all_preds, model_sql_path, model_record_path)

    print(f"Saved test set SQL results to {model_sql_path}")
    print(f"Saved test set record matches to {model_record_path}")

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_em, dev_record_f1, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
