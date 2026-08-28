#!/usr/bin/env python3
"""Focused tests for v0.5-B training resume support.

These exercise the real helpers from scripts/train_v05.py (save_ckpt,
latest_step_ckpt, restore_checkpoint) against a temporary checkpoint directory so
the committed checkpoints/v0_5/ artifacts are never touched.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import train_v05 as tv  # noqa: E402

# Load the real committed tokenizer + config once, BEFORE any CKPT_DIR patch,
# so build_tokenizer finds checkpoints/v0_5/tokenizer.json via the real path.
TOK = tv.build_tokenizer()
CFG = tv.load_cfg()
MODEL_CFG = CFG["model"]
TRAIN_CFG = CFG["training"]
ASSIST_ID = TOK.token_to_id(tv.ASSIST_START)


def _make_model_and_trainer(device):
    model = tv.build_model(TOK, MODEL_CFG, MODEL_CFG["max_seq_len"])
    trainer = tv.make_trainer(model, TRAIN_CFG, TOK, ASSIST_ID, MODEL_CFG)
    model.to(device)
    return model, trainer


def _op_state_copy(trainer):
    return {k: v.clone() if torch.is_tensor(v) else v
            for k, v in trainer.optimizer.state_dict().items()}


def _sched_state_copy(trainer):
    return trainer.scheduler.state_dict()


def _states_equal(a, b):
    """Deep equality for optimizer/scheduler state dicts (tensor-safe)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_states_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_states_equal(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape and torch.equal(a, b)
    return a == b


def _param(name, model):
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise KeyError(name)


@pytest.fixture
def iso(monkeypatch, tmp_path):
    """Point CKPT_DIR at an isolated temp directory for save/resume tests."""
    monkeypatch.setattr(tv, "CKPT_DIR", Path(tmp_path))
    monkeypatch.setattr(tv, "DATA_DIR", Path(tmp_path))
    return Path(tmp_path)


def test_checkpoint_contains_full_resume_state(iso):
    device = torch.device("cpu")
    model, trainer = _make_model_and_trainer(device)
    trainer.global_step = 250
    trainer.training_history = [{"step": 250, "loss": 4.5}]
    path = iso / "checkpoint_step_250.pt"
    tv.save_ckpt(model, trainer, str(path))

    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert set(saved.keys()) >= {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "global_step", "training_history", "config"}
    assert saved["global_step"] == 250
    assert saved["training_history"] == [{"step": 250, "loss": 4.5}]
    assert tv.ckpt_contains_required_state(path) is True


def test_latest_step_ckpt_picks_newest(iso):
    device = torch.device("cpu")
    model, trainer = _make_model_and_trainer(device)
    for step in (250, 500):
        trainer.global_step = step
        tv.save_ckpt(model, trainer, str(iso / f"checkpoint_step_{step}.pt"))

    latest = tv.latest_step_ckpt()
    assert latest is not None
    assert latest.name == "checkpoint_step_500.pt"
    # newer than both step files -> ignore it
    tv.save_ckpt(model, trainer, str(iso / "best_model.pt"))
    assert tv.latest_step_ckpt().name == "checkpoint_step_500.pt"


def test_no_checkpoint_returns_none(iso):
    assert tv.latest_step_ckpt() is None


def test_resume_restores_global_step_and_history(iso):
    device = torch.device("cpu")
    model, trainer = _make_model_and_trainer(device)
    trainer.global_step = 500
    trainer.training_history = [{"step": 500, "loss": 3.9}]
    tv.save_ckpt(model, trainer, str(iso / "checkpoint_step_500.pt"))

    model2, trainer2 = _make_model_and_trainer(device)
    assert trainer2.global_step == 0
    tv.restore_checkpoint(model2, trainer2, iso / "checkpoint_step_500.pt", device)
    assert trainer2.global_step == 500
    assert trainer2.training_history == [{"step": 500, "loss": 3.9}]


def test_resume_restores_weights(iso):
    device = torch.device("cpu")
    model, trainer = _make_model_and_trainer(device)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
    tv.save_ckpt(model, trainer, str(iso / "checkpoint_step_250.pt"))

    model2, trainer2 = _make_model_and_trainer(device)
    w0 = _param("token_embedding.weight", model).clone()
    w2 = _param("token_embedding.weight", model2).clone()
    assert not torch.equal(w0, w2)
    tv.restore_checkpoint(model2, trainer2, iso / "checkpoint_step_250.pt", device)
    assert torch.equal(_param("token_embedding.weight", model2), w0)


def test_resume_restores_optimizer_and_scheduler_state(iso):
    device = torch.device("cpu")
    model, trainer = _make_model_and_trainer(device)
    # Take a real forward/backward/step so AdamW momentum state is non-empty and
    # the scheduler advances -- only then is optimizer/scheduler state distinct
    # from a freshly initialized trainer and meaningful to restore.
    inp = torch.randint(0, TOK.get_vocab_size(), (2, 16), dtype=torch.long)
    tgt = inp.roll(-1, dims=1)
    loss = trainer._compute_loss(model(inp), inp, tgt)
    loss.backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.optimizer.zero_grad()

    tv.save_ckpt(model, trainer, str(iso / "checkpoint_step_250.pt"))
    opt_before = _op_state_copy(trainer)
    sched_before = _sched_state_copy(trainer)

    model2, trainer2 = _make_model_and_trainer(device)
    assert not _states_equal(trainer2.optimizer.state_dict(), opt_before)
    tv.restore_checkpoint(model2, trainer2, iso / "checkpoint_step_250.pt", device)
    assert _states_equal(trainer2.optimizer.state_dict(), opt_before)
    assert _states_equal(trainer2.scheduler.state_dict(), sched_before)


def test_second_run_continues_not_restarts(iso):
    """The second `train` invocation must start from the restored checkpoint's
    global_step rather than restart from 0."""
    device = torch.device("cpu")
    # Simulated first run: reaches and saves a checkpoint at global_step 250.
    model, trainer = _make_model_and_trainer(device)
    trainer.global_step = 250
    tv.save_ckpt(model, trainer, str(iso / "checkpoint_step_250.pt"))

    # Simulated second run: fresh model + trainer, discovers the latest
    # checkpoint and restores from it (exactly like cmd_train does).
    model2, trainer2 = _make_model_and_trainer(device)
    assert trainer2.global_step == 0  # fresh state before resume
    latest = tv.latest_step_ckpt()
    assert latest is not None
    assert latest.name == "checkpoint_step_250.pt"
    tv.restore_checkpoint(model2, trainer2, latest, device)

    # The resumed run must continue from 250, never reset to 0.
    assert trainer2.global_step == 250
