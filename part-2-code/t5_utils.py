import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    if not getattr(args, "use_wandb", False):
        return None

    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping wandb setup.")
        return None

    project = getattr(args, "wandb_project", "t5_nl2sql")
    run_name = getattr(args, "wandb_run_name", None)

    wandb.init(
        project=project,
        name=run_name,
        config=vars(args),
    )
    return wandb

def initialize_model(args): 
    model_name = getattr(args, "model_name_or_path", "google-t5/t5-small")

    init_from_scratch = getattr(args, "init_from_scratch", False)
    # (optional alternative flag)
    if hasattr(args, "init_method") and args.init_method == "scratch":
        init_from_scratch = True

    if init_from_scratch:
        # Same architecture as google-t5/t5-small, but random weights
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)
        print("Initialized T5 model from scratch with config:", model_name)
    else:
        # Load pretrained weights
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        print("Loaded pretrained T5 model from:", model_name)

    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    filename = "checkpoint-best.pt" if best else "checkpoint-last.pt"
    ckpt_path = os.path.join(checkpoint_dir, filename)

    # You can add more things to this dict if needed (optimizer, scheduler, epoch, etc.)
    state = {
        "model_state_dict": model.state_dict(),
    }

    torch.save(state, ckpt_path)
    print(f"Saved model checkpoint to {ckpt_path}")

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    checkpoint_dir = getattr(args, "checkpoint_dir", "checkpoints")
    filename = "checkpoint-best.pt" if best else "checkpoint-last.pt"
    ckpt_path = os.path.join(checkpoint_dir, filename)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # Initialize a fresh model (same architecture) first
    model = initialize_model(args)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Loaded model checkpoint from {ckpt_path}")
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

