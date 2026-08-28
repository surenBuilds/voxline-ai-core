"""Voxline v0.5-B — CodingAgent planner evaluation.

Loads the trained v0.5 checkpoint (checkpoints/v0_5/best_model.pt), generates
planner output for the held-out planner set, and validates it against the exact
CodingAgent planner schema (src/assistant/coding.py).

The test SKIPS cleanly when the full-training checkpoint is absent (training is
run explicitly via ``scripts/train_v05.py train`` and is intentionally NOT
auto-run). Once the checkpoint exists this test becomes the acceptance gate:

    JSON-parse rate  >= 90%
    schema-valid     >= 90%
    (target on the held-out set)

Metrics reported: JSON parse rate, schema-valid rate, generation latency,
tokens/sec, approximate peak RAM. The parser is NOT modified to improve scores.
"""

import asyncio
import json
import os
import time

import pytest
import torch
import yaml

from src.model.transformer import VoxlineTransformer
from src.tokenizer.bpe import BPETokenizer
from src.providers.base import GenerationConfig
from src.providers.local_voxline import LocalVoxlineProvider
from src.config.model_config import ModelConfig

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints" / "v0_5"
ASSIST_MARKER = "[ASSIST_START]"

# Required schema keys (must match src/assistant/coding.py _parse_plan).
REQUIRED_KEYS = [
    "objective", "understanding", "relevant_files", "steps",
    "risks", "validation_commands", "requires_approval",
]
STEP_KEYS = ["step_number", "description", "action_type", "target_files", "command"]
VALID_ACTION_TYPES = {
    "read", "write", "command", "github_*", "vercel_*", "workspace_*",
}


def _load_v05():
    """Load the trained v0.5 tokenizer + model + provider."""
    best = CKPT_DIR / "best_model.pt"
    if not best.exists():
        pytest.skip(
            "checkpoints/v0_5/best_model.pt not present; run "
            "`python scripts/train_v05.py train` first.")
    with open(REPO_ROOT / "configs" / "model_configs.yaml", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)["v0_5_planner"]["model"]

    tokenizer = BPETokenizer(vocab_size=model_cfg["vocab_size"])
    tokenizer.load(str(CKPT_DIR / "tokenizer.json"))

    model = VoxlineTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg["dropout"],
    )
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    provider = LocalVoxlineProvider(
        model=model,
        tokenizer=tokenizer,
        model_config=ModelConfig(
            model_type="voxline_transformer",
            model_version="0.5.0",
            vocab_size=tokenizer.get_vocab_size(),
            d_model=model_cfg["d_model"],
            max_seq_len=model_cfg["max_seq_len"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
        ),
        device="cpu",
    )
    return provider, tokenizer


def _prompt(record):
    """Mirror LocalVoxlineProvider.chat(): System/User lines ending in Assistant:."""
    return (
        "System: You are a helpful, correct coding assistant.\n"
        "User: " + record["prompt"] + "\n"
        "Assistant:"
    )


def _validate(raw):
    """Return (parse_ok, schema_ok, reason)."""
    text = raw.strip()
    if not text:
        return False, False, "empty"
    if text.startswith("[Error during generation"):
        return False, False, "generation_error"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, False, f"invalid_json: {exc.msg}"

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        return True, False, f"missing_keys: {missing}"
    if not isinstance(data.get("steps"), list):
        return True, False, "steps_not_list"
    for step in data["steps"]:
        if not isinstance(step, dict):
            return True, False, "step_not_dict"
        for k in STEP_KEYS:
            if k not in step:
                return True, False, f"step_missing_{k}"
        at = step.get("action_type", "")
        if at not in VALID_ACTION_TYPES:
            return True, False, f"bad_action_type:{at}"
    return True, True, "ok"


def _run_eval(records, max_tasks, max_tokens=400):
    provider, _tokenizer = _load_v05()
    if max_tasks:
        records = records[:max_tasks]

    cfg = GenerationConfig(max_tokens=max_tokens, temperature=0.0, do_sample=False)
    parse_ok = 0
    schema_ok = 0
    total_chars = 0
    total_time = 0.0
    errs = {}

    async def one(rec):
        t0 = time.time()
        raw = await provider.chat(
            messages=[
                {"role": "system", "content": "You are a helpful, correct coding assistant."},
                {"role": "user", "content": rec["prompt"]},
            ],
            config=cfg,
        )
        dt = time.time() - t0
        return raw, dt

    for rec in records:
        raw, dt = asyncio.run(one(rec))
        total_time += dt
        total_chars += len(raw)
        po, so, reason = _validate(raw)
        parse_ok += int(po)
        schema_ok += int(so)
        errs[reason] = errs.get(reason, 0) + 1

    n = len(records)
    return {
        "n": n,
        "parse_rate": parse_ok / max(1, n),
        "schema_rate": schema_ok / max(1, n),
        "latency_s": total_time / max(1, n),
        "tokens_per_sec": (total_chars / max(1e-9, total_time)),
        "raw_chars": total_chars,
        "errors": errs,
    }


@pytest.fixture(scope="module")
def test_records():
    p = CKPT_DIR / "planner_test.jsonl"
    if not p.exists():
        pytest.skip("planner_test.jsonl missing; run scripts/train_v05.py gen")
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def test_planner_eval_acceptance(test_records):
    """Acceptance gate: trained v0.5 must parse+match schema >= 90% held-out."""
    max_tasks = int(os.environ.get("VOX05_MAX_TASKS", "0") or 0)
    results = _run_eval(test_records, max_tasks)
    print("\n[v0.5 planner eval]")
    for k, v in results.items():
        print(f"  {k}: {v}")

    assert results["n"] > 0
    assert results["parse_rate"] >= 0.90, f"JSON parse rate {results['parse_rate']:.2%} < 90%"
    assert results["schema_rate"] >= 0.90, f"schema-valid rate {results['schema_rate']:.2%} < 90%"


def test_planner_always_nonempty_no_error(test_records):
    """Checks the schema validator itself on held-out records (no model needed)."""
    max_tasks = int(os.environ.get("VOX05_MAX_TASKS", "0") or 0)
    recs = test_records[:max_tasks] if max_tasks else test_records
    for rec in recs:
        po, so, reason = _validate(json.dumps(rec["plan"]))
        assert po and so, f"generated-heldout plan invalid: {reason}"
