"""
Bangla SLM Pretraining Script 

WHAT THIS SCRIPT DOES, IN PLAIN WORDS:
  1. Downloads your training data (train.bin, val.bin, tokenizer) from
     Hugging Face Hub if it isn't already sitting on disk.
  2. Loads that data using a memory-mapped dataset (so we never load the
     whole file into RAM at once).
  3. Builds the BanglaSLM model (defined in slm_model.py).
  4. Runs a training loop: read a batch -> compute loss -> update weights.
  5. Every so often, checks how the model is doing on validation data,
     prints a sample of generated text, and saves a checkpoint to disk.
  6. If the script is interrupted and restarted, it automatically resumes
     from the last checkpoint instead of starting over.

HOW TO RUN THIS:
    python train_slm.py

  All settings below have sensible defaults, so running it with no
  extra arguments will "just work" as long as you have a GPU available.
  You can override any setting from the command line, for example:
    python train_slm.py --max-steps 20 --eval-every 10 --save-every 10

REQUIREMENTS (install these first):
    pip install torch sentencepiece numpy huggingface_hub
"""

import argparse #built in Python module used for CLI, values can be modified from CLI
import math
import os
import sys
import time

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset

# This imports the model architecture from the OTHER file in this project,
# slm_model.py. That file must be sitting in the same folder as this
# script. We are not "running" slm_model.py here -- we are just importing
# two things that are defined inside it: the BanglaSLM class (the model
# itself) and the SLMConfig class (a small container that holds settings
# like how many layers the model has).
from slm_model import BanglaSLM, SLMConfig


# =====================================================================
# STEP 1: DEFINE ALL THE SETTINGS (COMMAND-LINE ARGUMENTS)
# =====================================================================
# Instead of hardcoding numbers and paths directly in the code, we collect
# them here. This means you can change things like the batch size or the
# number of training steps WITHOUT editing the code -- you just pass a
# different value when you run the script from the terminal.

def parse_args():
    parser = argparse.ArgumentParser(description="Bangla SLM pretraining")

    # ---- Where the training data lives ----
    # If these files are not found locally, the script will try to
    # download them automatically from Hugging Face Hub (see Step 2).
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Local folder where train.bin, val.bin, and the tokenizer live "
             "(or will be downloaded to).",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="./checkpoints",
        help="Local folder where model checkpoints will be saved.",
    )

    # ---- Hugging Face Hub settings, for auto-downloading the dataset ----
    parser.add_argument(
        "--hf-repo-id",
        default="faria00/bangla-slm-pretrain-data",
        help="The Hugging Face dataset repo to download train.bin, val.bin, "
             "and the tokenizer from, if they are not already present locally.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        default=False,
        help="If set, never try to download from Hugging Face Hub, even if "
             "the local files are missing. Useful if you already placed the "
             "files by hand.",
    )

    # ---- Model architecture settings ----
    # These control the SIZE of the model. Changing these means you are
    # training a different-sized model, not just changing training speed.
    parser.add_argument("--n-layer", type=int, default=8,
                         help="Number of transformer layers stacked on top of each other.")
    parser.add_argument("--n-head", type=int, default=6,
                         help="Number of attention heads per layer.")
    parser.add_argument("--n-embd", type=int, default=384,
                         help="Size of the embedding vector for each token.")
    parser.add_argument("--block-size", type=int, default=512,
                         help="How many tokens of context the model can see at once.")

    # ---- Training hyperparameters ----
    #we will be taking 512 tokens in one batch with each token represented in 384 dim vector. 
    # and total we have 32 batches so its 512x32 tokens [32, 512, 384]
    parser.add_argument("--micro-batch", type=int, default=32,
                         help="How many sequences we process in one forward/backward pass. "
                              "Lower this if you run out of GPU memory.")
    parser.add_argument("--grad-accum", type=int, default=16,
                         help="How many micro-batches we accumulate gradients over before "
                              "actually updating the model weights. This lets us simulate a "
                              "bigger batch size than what fits in GPU memory at once.")
    parser.add_argument("--max-steps", type=int, default=20_000,
                         help="Total number of optimizer update steps to train for.")
    parser.add_argument("--warmup-steps", type=int, default=400,
                         help="Number of steps at the start where the learning rate "
                              "ramps up gradually, instead of starting at full strength.")
    parser.add_argument("--lr", type=float, default=6e-4,
                         help="The highest learning rate reached, after warmup.")
    parser.add_argument("--min-lr", type=float, default=6e-5,
                         help="The lowest learning rate, reached at the very end of training.")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                         help="A regularization setting that discourages very large weights.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                         help="Maximum allowed size of the gradient, to prevent unstable updates.")

    # ---- How often to log / evaluate / save ----
    parser.add_argument("--eval-every", type=int, default=500,
                         help="Run a validation check every N steps.")
    parser.add_argument("--eval-batches", type=int, default=60,
                         help="How many validation batches to average over each time we evaluate.")
    parser.add_argument("--save-every", type=int, default=500,
                         help="Save a checkpoint to disk every N steps.")
    parser.add_argument("--log-every", type=int, default=20,
                         help="Print a training progress line every N steps.")

    # ---- Misc system settings ----
    parser.add_argument("--compile", action="store_true", default=False,
                         help="If set, use torch.compile to speed up training. "
                              "Requires a working C compiler on your machine; leave "
                              "this off if you're not sure.")
    parser.add_argument("--num-workers", type=int, default=0,
                         help="Number of background processes used to load data. "
                              "0 means the main process loads it, which is the right "
                              "choice while --preload is on: slicing an array that is "
                              "already in RAM is so cheap that worker processes would "
                              "only add startup cost and duplicate the data in memory.")
    parser.add_argument("--no-preload", dest="preload", action="store_false", default=True,
                         help="Memory-map the token files from disk instead of reading "
                              "them into RAM. Only use this if train.bin is too large to "
                              "fit in RAM. On a spinning hard drive, memory-mapping with "
                              "shuffling can crash the run with STATUS_IN_PAGE_ERROR.")
    parser.add_argument("--seed", type=int, default=1337,
                         help="Random seed, for reproducibility.")

    return parser.parse_args()


# =====================================================================
# STEP 2: MAKE SURE THE DATA IS ON DISK (DOWNLOAD FROM HF HUB IF NEEDED)
# =====================================================================

def ensure_data_is_available(data_dir, hf_repo_id, skip_download):
    """
    Checks whether train.bin, val.bin, and the tokenizer already exist in
    data_dir. If any of them are missing, and skip_download is False, we
    download them from the given Hugging Face dataset repo.
    """

    train_bin_path = os.path.join(data_dir, "train.bin")
    val_bin_path = os.path.join(data_dir, "val.bin")
    tokenizer_path = os.path.join(data_dir, "tokenizer", "bangla_sp.model")

    all_files_present = (
        os.path.exists(train_bin_path)
        and os.path.exists(val_bin_path)
        and os.path.exists(tokenizer_path)
    )

    if all_files_present:
        print("Found train.bin, val.bin, and tokenizer already on disk. Skipping download.")
        return train_bin_path, val_bin_path, tokenizer_path

    if skip_download:
        # The user explicitly told us not to download, but the files are
        # missing. We cannot continue, so we stop with a clear error
        # message instead of a confusing crash later on.
        raise FileNotFoundError(
            "Data files are missing locally and --skip-download was set, "
            "so we cannot fetch them automatically. Please place train.bin, "
            "val.bin, and tokenizer/bangla_sp.model into: " + data_dir
        )

    print(f"Data files not found locally. Downloading from Hugging Face Hub "
          f"repo '{hf_repo_id}' ...")

    # hf_hub_download fetches ONE file at a time from a Hugging Face repo
    # and returns the local path where it was saved. We import it here
    # (rather than at the top of the file) so that people who already have
    # their data locally and use --skip-download don't need this library
    # installed at all.
    from huggingface_hub import hf_hub_download

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "tokenizer"), exist_ok=True)

    downloaded_train_bin = hf_hub_download(
        repo_id=hf_repo_id,
        filename="train.bin",
        repo_type="dataset",
        local_dir=data_dir,
    )
    downloaded_val_bin = hf_hub_download(
        repo_id=hf_repo_id,
        filename="val.bin",
        repo_type="dataset",
        local_dir=data_dir,
    )
    downloaded_tokenizer = hf_hub_download(
        repo_id=hf_repo_id,
        filename="bangla_sp.model",
        repo_type="dataset",
        local_dir=os.path.join(data_dir, "tokenizer"),
    )

    print("Download complete.")
    return downloaded_train_bin, downloaded_val_bin, downloaded_tokenizer


# =====================================================================
# STEP 3: THE DATASET CLASS
# =====================================================================
# A PyTorch "Dataset" is just an object that knows two things:
#    (a) how many training examples exist (the __len__ method)
#    (b) how to fetch example number i (the __getitem__ method)
# PyTorch's DataLoader uses these two methods to serve us shuffled batches.

class BinaryMemoryMapDataset(Dataset):
    """
    Our tokenized text is stored as one giant binary file of integers
    (token IDs), NOT as a normal text file. This class reads small slices
    out of that giant file on demand, using "memory mapping" -- meaning
    the operating system handles paging data in from disk as needed,
    instead of us loading the entire multi-hundred-megabyte file into
    RAM up front.
    """

    def __init__(self, file_path, block_size, dtype=np.uint16, preload=True):
        self.file_path = file_path
        self.block_size = block_size   # how many tokens go into one training example
        self.dtype = dtype             # the numeric type each token ID is stored as
        self.preload = preload

        # Figure out how many total tokens are in the file, just from its
        # size in bytes divided by how many bytes each token takes up.
        bytes_per_token = np.dtype(dtype).itemsize
        self.total_tokens = os.path.getsize(file_path) // bytes_per_token

        # Each training example needs block_size + 1 tokens (we'll explain
        # why in __getitem__ below), so we figure out how many
        # non-overlapping examples fit in the file.
        self.num_samples = (self.total_tokens - 1) // block_size

        # There are two ways to read the token file, and the right choice
        # depends on what kind of drive the file is sitting on.
        #
        # preload=True (the default): read the WHOLE file into RAM once, in
        #   a single sequential pass. This is the safe option on a spinning
        #   hard drive. Memory-mapping instead asks the operating system to
        #   fetch a kilobyte here and a kilobyte there at random across
        #   hundreds of megabytes, and a mechanical drive must physically
        #   move its read head for every one of those requests. If a single
        #   page-in stalls or fails, Windows kills the process outright with
        #   STATUS_IN_PAGE_ERROR (exception code 0xc0000006). Reading the
        #   file sequentially up front avoids that failure mode entirely,
        #   and every batch afterwards is served from RAM, which is far
        #   faster than any disk.
        #
        # preload=False: memory-map the file and let the OS page it in on
        #   demand. Use this only when the file is too large to fit in RAM,
        #   and preferably only when it lives on an SSD.
        self.token_array = None
        self.memory_map = None

        if self.preload:
            size_in_mb = os.path.getsize(file_path) / 1e6
            print(f"Loading {os.path.basename(file_path)} into RAM ({size_in_mb:.0f} MB)...")
            self.token_array = np.fromfile(file_path, dtype=dtype)

    def __len__(self):
        # Tells PyTorch how many examples this dataset has in total.
        return self.num_samples

    def __getitem__(self, index):
        if self.preload:
            # Already sitting in RAM -- nothing to open, nothing to fault in.
            source_array = self.token_array
        else:
            # Open the memory-mapped file on first use. We do this lazily
            # rather than in __init__ because PyTorch's DataLoader can spawn
            # separate worker processes, and each one needs its own
            # independent file handle.
            if self.memory_map is None:
                self.memory_map = np.memmap(self.file_path, dtype=self.dtype, mode="r")
            source_array = self.memory_map

        # Work out which slice of the giant token array belongs to this
        # particular training example.
        start_position = index * self.block_size
        end_position = start_position + self.block_size + 1

        # Pull out block_size + 1 tokens. We need one extra token because
        # of how "next token prediction" works: given tokens [0..N-1] as
        # input, the model tries to predict tokens [1..N] as the target.
        # So input and target are the SAME sequence, just shifted by one
        # position.
        raw_slice = source_array[start_position:end_position]
        token_chunk = torch.from_numpy(np.array(raw_slice, dtype=np.int64))

        input_tokens = token_chunk[:-1]   # everything except the last token
        target_tokens = token_chunk[1:]   # everything except the first token

        return input_tokens, target_tokens


def infinite_data_generator(dataloader):
    """
    A normal DataLoader stops once it has gone through the whole dataset
    one time (one "epoch"). Our training loop is written in terms of
    total STEPS, not epochs, so this helper just loops the DataLoader
    forever, restarting from the beginning each time it runs out.
    """
    while True:
        for batch in dataloader:
            yield batch


# =====================================================================
# STEP 4: THE MAIN TRAINING FUNCTION
# =====================================================================

def main():
    args = parse_args()

    # Windows terminals default to the cp1252 encoding, which cannot represent
    # Bangla characters. Without this, printing a generated sample raises
    # UnicodeEncodeError and kills the run -- right between the evaluation and
    # the checkpoint save, which is the worst possible moment to crash.
    # errors="replace" means an unprintable character shows as "?" instead of
    # bringing down hours of training.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ---- 4a. Make sure the data files exist, downloading if necessary ----
    train_bin_path, val_bin_path, tokenizer_path = ensure_data_is_available(
        data_dir=args.data_dir,
        hf_repo_id=args.hf_repo_id,
        skip_download=args.skip_download,
    )

    # ---- 4b. Basic setup ----
    os.makedirs(args.ckpt_dir, exist_ok=True)

    torch.manual_seed(args.seed)  # makes results reproducible across runs
    torch.backends.cuda.matmul.allow_tf32 = True   # small speed boost on modern GPUs
    torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        device = "cuda"
    else:
        # We are assuming GPU training. If no GPU is found, we stop here
        # with a clear message rather than silently training on CPU,
        # which would be extremely slow for this model size.
        raise SystemError(
            "No CUDA GPU was detected. This script is meant to be run with "
            "a GPU (for example, in Colab with a GPU runtime selected, or "
            "on a machine with an NVIDIA GPU and CUDA installed)."
        )

    # Newer GPUs support "bfloat16" mixed precision, which is more
    # numerically stable than the older "float16". We check which one
    # this GPU supports and use the better option automatically.
    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    print(f"Using device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Mixed precision dtype: {amp_dtype}")

    # ---- 4c. Build the datasets and data loaders ----
    train_dataset = BinaryMemoryMapDataset(train_bin_path, args.block_size, preload=args.preload)
    val_dataset = BinaryMemoryMapDataset(val_bin_path, args.block_size, preload=args.preload)

    # Load the tokenizer so we know the vocabulary size, and so we can
    # decode generated token IDs back into readable Bangla text later.
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.get_piece_size()

    tokens_per_optimizer_step = args.micro_batch * args.grad_accum * args.block_size
    print(f"Train set contains : {train_dataset.total_tokens / 1e6:.2f} million tokens")
    print(f"Val set contains   : {val_dataset.total_tokens / 1e6:.2f} million tokens")
    print(f"Each optimizer step consumes: {tokens_per_optimizer_step / 1e3:.1f}k tokens")
    total_token_budget = args.max_steps * tokens_per_optimizer_step
    print(f"Total training budget: {total_token_budget / 1e6:.1f} million tokens")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    train_iterator = infinite_data_generator(train_loader)

    # ---- 4d. Build the model ----
    model_config = SLMConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
    )
    model = BanglaSLM(model_config).to(device)

    total_params_millions = model.num_params() / 1e6
    non_embedding_params_millions = model.num_params(non_embedding=True) / 1e6
    print(f"Model has {total_params_millions:.2f}M total parameters "
          f"({non_embedding_params_millions:.2f}M excluding the embedding table)")

    # The optimizer is the algorithm that actually updates the model's
    # weights based on the gradients computed during backpropagation.
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, device_type=device)

    # The GradScaler helps prevent numerical underflow when training in
    # float16. It is not needed for bfloat16, so we only enable it when
    # amp_dtype is float16.
    use_grad_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device, enabled=use_grad_scaler)

    # We keep a reference to the "raw" (uncompiled) model. This is what
    # we save to checkpoints and use for text generation, because
    # torch.compile can wrap the model in a way that makes some
    # operations behave differently.
    raw_model = model
    if args.compile:
        print("Compiling model with torch.compile (this may take a minute)...")
        model = torch.compile(model)

    def get_learning_rate_for_step(step):
        """
        Implements a 'warmup then cosine decay' learning rate schedule:
          - For the first `warmup_steps`, the learning rate ramps up
            linearly from 0 to args.lr. This avoids destabilizing the
            model with a large learning rate before it has adjusted from
            its random initialization.
          - After warmup, the learning rate smoothly decreases from
            args.lr down to args.min_lr following a cosine curve, all the
            way to the end of training.
        """
        if step < args.warmup_steps:
            fraction_through_warmup = (step + 1) / args.warmup_steps
            return args.lr * fraction_through_warmup

        steps_after_warmup = step - args.warmup_steps
        total_decay_steps = max(1, args.max_steps - args.warmup_steps)
        progress = steps_after_warmup / total_decay_steps
        if progress > 1.0:
            progress = 1.0

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_range = args.lr - args.min_lr
        return args.min_lr + cosine_factor * lr_range

    # ---- 4e. Checkpoint saving and resuming ----
    last_checkpoint_path = os.path.join(args.ckpt_dir, "last.pt")
    best_checkpoint_path = os.path.join(args.ckpt_dir, "best.pt")

    def save_checkpoint(filepath, step, best_val_loss):
        """
        Saves everything needed to resume training later: the model
        weights, the optimizer's internal state, the scaler's state, and
        which step we were on.

        We save to a temporary file first and then rename it into place.
        Renaming a file is atomic on most filesystems, meaning it either
        fully happens or doesn't happen at all -- so if the machine
        crashes or disconnects DURING the save, we're left with either
        the old checkpoint or the new one, never a half-written corrupt
        file.
        """
        temp_filepath = filepath + ".tmp"
        checkpoint_contents = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "config": model_config.__dict__,
        }
        torch.save(checkpoint_contents, temp_filepath)
        os.replace(temp_filepath, filepath)

    start_step = 0
    best_val_loss = float("inf")

    if os.path.exists(last_checkpoint_path):
        print("Found an existing checkpoint. Resuming training from it...")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resuming from step {start_step}. Best validation loss so far: {best_val_loss:.4f}")
    else:
        print("No existing checkpoint found. Starting training from scratch.")

    # ---- 4f. Helper functions for evaluation and sampling ----

    @torch.no_grad()  # disables gradient tracking, since we're not training here
    def evaluate_on_validation_set():
        """
        Runs the model on a handful of validation batches (data it was
        NOT trained on) and returns the average loss. This tells us how
        well the model generalizes, as opposed to just memorizing the
        training data.
        """
        model.eval()  # switches off dropout etc. (not used here, but good practice)
        collected_losses = []

        for batch_index, (inputs, targets) in enumerate(val_loader):
            if batch_index >= args.eval_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device, dtype=amp_dtype):
                _, loss = model(inputs, targets)

            collected_losses.append(loss.item())

        model.train()  # switch back to training mode before returning
        return float(np.mean(collected_losses))

    @torch.no_grad()
    def generate_sample_text(prompt="বাংলাদেশের", max_new_tokens=80):
        """
        Generates a short piece of text starting from `prompt`, so we can
        eyeball whether the model's output looks like real Bangla or
        gibberish. This is a quick sanity check, not a formal metric.
        """
        prompt_token_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([prompt_token_ids], device=device)

        generated_token_ids = raw_model.generate(
            input_tensor, max_new_tokens, temperature=0.8, top_k=50
        )
        model.train()

        return tokenizer.decode(generated_token_ids[0].tolist())

    # ---- 4g. The main training loop ----
    print("\n--- Starting training loop ---\n")
    model.train()
    time_of_last_log = time.time()

    for step in range(start_step, args.max_steps):

        # Update the learning rate for this step according to our schedule.
        current_lr = get_learning_rate_for_step(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss_for_this_step = 0.0

        # Gradient accumulation: instead of updating the weights after
        # every single micro-batch, we run several micro-batches, add up
        # (accumulate) their gradients, and only THEN update the weights
        # once. This lets us simulate a much larger batch size than would
        # otherwise fit in GPU memory.
        for _ in range(args.grad_accum):
            inputs, targets = next(train_iterator)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # autocast automatically runs the forward pass using the
            # faster mixed-precision dtype where it's safe to do so.
            with torch.autocast(device, dtype=amp_dtype):
                _, loss = model(inputs, targets)
                # We divide the loss by grad_accum so that the SUM of
                # gradients across all micro-batches ends up being
                # equivalent to averaging over the full effective batch.
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accumulated_loss_for_this_step += loss.item()

        # Before clipping gradients, we need to "unscale" them (undo the
        # scaling the GradScaler applied for numerical stability).
        scaler.unscale_(optimizer)

        # Gradient clipping prevents any single update from being too
        # large, which helps keep training stable.
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # ---- Logging ----
        if step % args.log_every == 0:
            elapsed_seconds = time.time() - time_of_last_log

            if step > start_step and elapsed_seconds > 0:
                tokens_per_second = (args.log_every * tokens_per_optimizer_step) / elapsed_seconds
            else:
                tokens_per_second = 0.0

            if accumulated_loss_for_this_step < 20:
                perplexity = math.exp(accumulated_loss_for_this_step)
            else:
                # exp() of a large loss would overflow, so we just report infinity
                perplexity = float("inf")

            print(
                f"Step {step:6d} | "
                f"Loss: {accumulated_loss_for_this_step:.4f} | "
                f"Perplexity: {perplexity:8.1f} | "
                f"Learning rate: {current_lr:.2e} | "
                f"Gradient norm: {grad_norm:.2f} | "
                f"Speed: {tokens_per_second / 1e3:.1f}k tokens/sec"
            )
            time_of_last_log = time.time()

        # ---- Periodic evaluation ----
        if step > 0 and step % args.eval_every == 0:
            validation_loss = evaluate_on_validation_set()

            if validation_loss < 20:
                validation_perplexity = math.exp(validation_loss)
            else:
                validation_perplexity = float("inf")

            sample_text = generate_sample_text()

            print(f"\n[EVALUATION] Step {step} | "
                  f"Validation loss: {validation_loss:.4f} | "
                  f"Validation perplexity: {validation_perplexity:.1f}")
            print(f"[SAMPLE OUTPUT] {sample_text[:150]}...\n")

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                save_checkpoint(best_checkpoint_path, step, best_val_loss)
                print(" -> This is the best model so far. Saved to best.pt\n")

            time_of_last_log = time.time()

        # ---- Periodic checkpoint saving ----
        if step > 0 and step % args.save_every == 0:
            save_checkpoint(last_checkpoint_path, step, best_val_loss)

    # Always save a final checkpoint once training finishes.
    save_checkpoint(last_checkpoint_path, args.max_steps - 1, best_val_loss)
    print("Training complete!")


if __name__ == "__main__":
    main()