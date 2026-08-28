#!/usr/bin/env python3
"""Voxline v0.5-B — Coding-Agent planner training pipeline.

Subcommands:
    gen      build the deterministic offline planner dataset/corpus
    sanity   run pre-training sanity checks (instantiate, 1024-token fwd/bwd,
             one optimizer step, save/load, memory)
    overfit  tiny 2-5 batch overfit test; measures per-step time and RAM
    train    full training run (NOT auto-run; execute explicitly)

All fully local. No external LLM at any point.
"""

import argparse
import json
import sys
import time
import yaml
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.transformer import VoxlineTransformer
from src.tokenizer.bpe import BPETokenizer
from src.training.trainer import TrainingConfig, Trainer, collate_batch

ASSIST_START = "[ASSIST_START]"
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<CLS>", "<SEP>", ASSIST_START]

DATA_DIR = ROOT / "data" / "v0_5"
CKPT_DIR = ROOT / "checkpoints" / "v0_5"


def load_cfg():
    with open(ROOT / "configs" / "model_configs.yaml") as f:
        all_cfg = yaml.safe_load(f)
    return all_cfg["v0_5_planner"]


def build_tokenizer(force_retrain=False):
    """Build or load the v0.5 BPE tokenizer with the assist sentinel reserved."""
    tok_path = CKPT_DIR / "tokenizer.json"
    model_cfg = load_cfg()["model"]
    if tok_path.exists() and not force_retrain:
        tok = BPETokenizer(vocab_size=model_cfg["vocab_size"],
                           special_tokens=SPECIAL_TOKENS)
        tok.load(str(tok_path))
        return tok
    corpus = DATA_DIR / "tokenizer_corpus.txt"
    if not corpus.exists():
        raise FileNotFoundError(
            "tokenizer corpus missing; run: python scripts/train_v05.py gen")
    with open(corpus, encoding="utf-8") as f:
        texts = [l.strip() for l in f if l.strip()]
    tok = BPETokenizer(vocab_size=model_cfg["vocab_size"],
                       special_tokens=SPECIAL_TOKENS)
    tok.fit(texts, num_merges=load_cfg()["training"].get("bpe_merges", 2500))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(str(tok_path))
    return tok


class PlannerDataset(torch.utils.data.Dataset):
    """LM dataset where the assistant response tail is isolated and the prompt
    tokens are masked to -100 in the target.

    Each corpus line has the form:
        System: ...\\nUser: <prompt>\\nAssistant: [ASSIST_START] {json}
    Construction of the mask:
      - Split each line at the literal ``ASSIST_START`` marker.
      - input  = encode(prompt) + [assist_id] + encode(response)
      - target = input shifted by one (next-token prediction).
      - Every target position at/before the assist marker's index is set to
        -100 (ignored by CrossEntropyLoss): the model only learns to predict the
        JSON response tokens, not to reproduce the instruction prompt.
    """
    def __init__(self, lines, tokenizer, assist_id, eos_id, max_seq_len):
        self.examples = []
        for line in lines:
            if not line.strip():
                continue
            if ASSIST_START in line:
                prompt, response = line.split(ASSIST_START, 1)
            else:
                prompt, response = line, ""
            toks = tokenizer.encode(prompt)
            toks.append(assist_id)
            toks.extend(tokenizer.encode(response))
            if eos_id is not None:
                toks.append(eos_id)
            toks = toks[:max_seq_len]
            if len(toks) < 2:
                continue
            inp = toks[:-1]
            tgt = list(toks[1:])
            # mask prompt tokens (positions before the assist marker in input)
            if assist_id in inp:
                ai = inp.index(assist_id)
            else:
                ai = len(inp)
            for j in range(ai):
                tgt[j] = -100
            self.examples.append((inp, tgt))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inp, tgt = self.examples[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def read_lines(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [l for l in (x.strip() for x in f) if l]


def build_model(tok, model_cfg, max_seq_len):
    model = VoxlineTransformer(
        vocab_size=tok.get_vocab_size(),
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        max_seq_len=max_seq_len,
        dropout=model_cfg["dropout"],
    )
    return model


def setup_device():
    return torch.device("cpu")


def make_trainer(model, training_cfg, tok, assist_id, model_cfg=None):
    if model_cfg is None:
        model_cfg = load_cfg()["model"]
    tcfg = TrainingConfig(
        vocab_size=model.vocab_size,
        d_model=model.d_model,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        d_ff=model_cfg.get("d_ff", 1024),
        max_seq_len=model.max_seq_len,
        dropout=model_cfg.get("dropout", 0.1),
        batch_size=training_cfg.get("batch_size", 4),
        learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
        warmup_steps=training_cfg.get("warmup_steps", 100),
        max_steps=training_cfg.get("max_steps", 3000),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        eval_steps=training_cfg.get("eval_steps", 200),
        save_steps=training_cfg.get("save_steps", 500),
        checkpoint_dir=str(CKPT_DIR),
        num_epochs=training_cfg.get("num_epochs", 40),
        patience=training_cfg.get("patience", 10),
        device="cpu",
    )
    return Trainer(model, tcfg, tokenizer=tok, assist_token_id=assist_id)


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def cmd_gen(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_v05_corpus as genmod
    nt, nv, nte, ntok, nlm = genmod.generate(args)
    print("Planner train:", nt)
    print("Planner val:", nv)
    print("Planner test(held-out):", nte)
    print("Tokenizer corpus lines:", ntok)
    print("Train LM corpus lines:", nlm)


def load_all():
    cfg = load_cfg()
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    tok = build_tokenizer()
    print("Actual vocab size:", tok.get_vocab_size())
    assist_id = tok.token_to_id(ASSIST_START)
    eos_id = tok.token_to_id("<EOS>")
    model = build_model(tok, model_cfg, model_cfg["max_seq_len"])
    train_lines = read_lines(DATA_DIR / "train_corpus.txt")
    val_lines = read_lines(DATA_DIR / "val_corpus.txt")
    train_ds = PlannerDataset(train_lines, tok, assist_id, eos_id,
                              model_cfg["max_seq_len"])
    val_ds = PlannerDataset(val_lines, tok, assist_id, eos_id,
                            model_cfg["max_seq_len"])
    return (model_cfg, training_cfg, tok, assist_id, model,
            train_ds, val_ds)


def loaders(train_ds, val_ds, batch_size):
    g = torch.Generator().manual_seed(42)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch,
        generator=g)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, val_loader


def ram_mb():
    """Current process RSS in MB, portable across Windows / POSIX."""
    import os
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            counter = PROCESS_MEMORY_COUNTERS()
            counter.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counter),
                ctypes.sizeof(counter))
            if ok and counter.WorkingSetSize:
                return counter.WorkingSetSize / 1e6
            return 0.0
        except Exception:
            return 0.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return float(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def cmd_sanity(args):
    model_cfg, training_cfg, tok, assist_id, model, train_ds, val_ds = load_all()
    print("== v0.5-B SANITY CHECKS ==")
    print("Model params:", f"{model.get_num_parameters():,}")
    print("max_seq_len:", model_cfg["max_seq_len"])
    print("assist_id:", assist_id, "(token:", ASSIST_START, ")")
    print("train examples:", len(train_ds), "| val examples:", len(val_ds))

    model.eval()
    device = setup_device()
    model.to(device)
    # 1024-token input forward
    xs = torch.randint(0, tok.get_vocab_size(), (1, model.max_seq_len),
                       dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(xs)
    print("1024-token forward OK:", tuple(out.shape))
    assert out.shape == (1, model.max_seq_len, tok.get_vocab_size())

    # backward + optimizer step on a real batch
    model.train()
    batch_size = training_cfg.get("batch_size", 4)
    train_loader, _ = loaders(train_ds, val_ds, batch_size)
    batch = next(iter(train_loader))
    inp, tgt = batch[0].to(device), batch[1].to(device)
    logits = model(inp)
    helper = Trainer(model, TrainingConfig(device="cpu"), tokenizer=tok,
                     assist_token_id=assist_id)
    loss = helper._compute_loss(logits, inp, tgt)
    print("Masked loss (first batch):", round(loss.item(), 4))
    t0 = time.time()
    loss.backward()
    helper.optimizer.step()
    helper.optimizer.zero_grad()
    print("Backward + one optimizer step OK in %.3fs" % (time.time() - t0))

    # save/load round-trip
    tmp = CKPT_DIR / "sanity_tmp.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "config": asdict_cfg(model_cfg)}, tmp)
    m2 = build_model(tok, model_cfg, model_cfg["max_seq_len"])
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    m2.load_state_dict(ck["model_state_dict"])
    tmp.unlink()
    print("Checkpoint save/load OK")

    print("Peak process RAM: %.1f MB" % ram_mb())
    print("SANITY: PASS")


def asdict_cfg(model_cfg):
    """Persist model config as plain dict (model is reinstantiable from it)."""
    return dict(model_cfg)


def cmd_overfit(args):
    model_cfg, training_cfg, tok, assist_id, model, train_ds, val_ds = load_all()
    device = setup_device()
    model.to(device)
    train_loader, _ = loaders(train_ds, val_ds, 4)
    # take ONE small fixed batch and re-run it repeatedly: a true memorization
    # check (the same examples over successive steps -> loss should fall to ~0).
    batch = next(iter(train_loader))
    inp, tgt = batch[0].to(device), batch[1].to(device)

    trainer = make_trainer(model, training_cfg, tok, assist_id, model_cfg)
    # Use full LR immediately (no warmup) so a few steps visibly memorize the
    # fixed batch; the real training run keeps warmup.
    trainer.config.warmup_steps = 0
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer,
        lambda step: 1.0)
    steps = max(2, args.steps)
    print(f"== TINY OVERFIT TEST: 1 fixed batch, {steps} optimizer steps, "
          f"block_size {model_cfg['max_seq_len']} ==")
    losses = []
    t0 = time.time()
    for step in range(steps):
        logits = model(inp)
        loss = trainer._compute_loss(logits, inp, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       training_cfg.get("max_grad_norm", 1.0))
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        losses.append(loss.item())
        print(f"  step {step + 1}: loss={loss.item():.4f}")
    elapsed = time.time() - t0
    print(f"Overfit elapsed: {elapsed:.2f}s over {steps} step(s) "
          f"= {elapsed/steps:.2f}s/step")
    print("Peak RAM: %.1f MB" % ram_mb())
    final = losses[-1] if losses else float("nan")
    print("Final overfit loss:", round(final, 4))
    reduced = steps >= 2 and losses[-1] < losses[0] * 0.5
    print("OVERFIT test: loss decreased (>50%):", reduced)
    if not reduced:
        print("WARNING: overfit loss did not decrease enough; inspect pipeline.")


def cmd_train(args):
    model_cfg, training_cfg, tok, assist_id, _, train_ds, val_ds = load_all()
    device = setup_device()
    batch_size = training_cfg.get("batch_size", 4)
    train_loader, val_loader = loaders(train_ds, val_ds, batch_size)

    model = build_model(tok, model_cfg, model_cfg["max_seq_len"])
    trainer = make_trainer(model, training_cfg, tok, assist_id, model_cfg)
    model.to(device)

    # Resume: continue from the latest checkpoint's global_step (never step 0).
    latest = latest_step_ckpt()
    if latest is not None:
        restore_checkpoint(model, trainer, latest, device)
        print(f"Resumed from {latest.name} at "
              f"global_step={trainer.global_step}")

    # --max-steps / config is the TOTAL target global step for this invocation,
    # not an increment. A run continues from global_step toward this target.
    target = args.max_steps if args.max_steps else trainer.config.max_steps
    if trainer.global_step >= target:
        print(f"Already at global_step={trainer.global_step} >= target={target}; "
              "nothing to do.")
        return
    print(f"Training from global_step={trainer.global_step} to target={target}...")
    t0 = time.time()
    running = 0.0
    best = float("inf")
    patience = trainer.config.patience
    no_improve = 0
    while trainer.global_step < target:
        model.train()
        for inp, tgt in train_loader:
            if trainer.global_step >= target:
                break
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)
            loss = trainer._compute_loss(logits, inp, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           trainer.config.max_grad_norm)
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.optimizer.zero_grad()
            running += loss.item()
            trainer.global_step += 1
            if trainer.global_step % trainer.config.eval_steps == 0:
                vloss = evaluate(trainer, model, val_loader, device)
                avg = running / trainer.config.eval_steps
                print(f"step {trainer.global_step}: train_loss={avg:.4f} "
                      f"val_loss={vloss:.4f} "
                      f"lr={trainer.scheduler.get_last_lr()[0]:.2e} "
                      f"({(time.time()-t0):.0f}s)")
                running = 0.0
                if vloss < best:
                    best = vloss
                    no_improve = 0
                    save_ckpt(model, trainer, f"{CKPT_DIR / 'best_model.pt'}")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("Early stopping.")
                        save_ckpt(model, trainer, f"{CKPT_DIR / 'best_model.pt'}")
                        save_ckpt(model, trainer,
                                  f"{CKPT_DIR / f'checkpoint_step_{trainer.global_step}.pt'}")
                        return
            if (trainer.global_step % trainer.config.save_steps == 0
                    or trainer.global_step == target):
                save_ckpt(model, trainer,
                          f"{CKPT_DIR / f'checkpoint_step_{trainer.global_step}.pt'}")
    save_ckpt(model, trainer, f"{CKPT_DIR / 'best_model.pt'}")
    save_ckpt(model, trainer,
              f"{CKPT_DIR / f'checkpoint_step_{trainer.global_step}.pt'}")
    print(f"Training complete in {(time.time()-t0)/60:.1f} min")


def evaluate(trainer, model, val_loader, device):
    model.eval()
    total = 0.0
    nb = 0
    with torch.no_grad():
        for inp, tgt in val_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)
            total += trainer._compute_loss(logits, inp, tgt).item()
            nb += 1
    model.train()
    return total / max(1, nb)


def save_ckpt(model, trainer, path):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "config": dict(trainer.config.__dict__ if hasattr(trainer.config, "__dict__")
                       else trainer.config),
        "global_step": trainer.global_step,
        "training_history": trainer.training_history,
    }, path)
    print("Saved:", path)


def latest_step_ckpt():
    """Return the path to the newest checkpoint_step_*.pt in CKPT_DIR, or None."""
    if not CKPT_DIR.exists():
        return None
    ckpts = []
    for p in CKPT_DIR.glob("checkpoint_step_*.pt"):
        try:
            n = int(p.stem.split("_")[-1])
            ckpts.append((n, p))
        except ValueError:
            continue
    if not ckpts:
        return None
    return max(ckpts, key=lambda x: x[0])[1]


def restore_checkpoint(model, trainer, path, device):
    """Load a full resume checkpoint into an existing model + trainer, restoring
    optimizer, scheduler, global_step and training history."""
    ck = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    trainer.optimizer.load_state_dict(ck["optimizer_state_dict"])
    trainer.scheduler.load_state_dict(ck["scheduler_state_dict"])
    trainer.global_step = ck.get("global_step", 0)
    trainer.training_history = ck.get("training_history", [])
    return ck


def ckpt_contains_required_state(path):
    """Return True if a checkpoint file has full resume state (used in tests)."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return all(k in saved for k in (
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "global_step", "training_history", "config"))


def main():
    ap = argparse.ArgumentParser(description="Voxline v0.5-B planner training")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--n-train", type=int, default=300)
    g.add_argument("--n-val", type=int, default=50)
    g.add_argument("--n-test", type=int, default=50)
    g.add_argument("--general-size", type=int, default=800)

    sp = sub.add_parser("sanity")

    op = sub.add_parser("overfit")
    op.add_argument("--steps", type=int, default=5)

    tp = sub.add_parser("train")
    tp.add_argument("--max-steps", type=int, default=0)

    args = ap.parse_args()
    if args.cmd == "gen":
        cmd_gen(args)
    elif args.cmd == "sanity":
        cmd_sanity(args)
    elif args.cmd == "overfit":
        cmd_overfit(args)
    elif args.cmd == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
