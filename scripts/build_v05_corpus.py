#!/usr/bin/env python3
"""Voxline v0.5 — Deterministic offline planner dataset / corpus generator.

Builds an instruction-following, JSON-planning dataset matching the existing
CodingAgent planner schema, WITHOUT calling any external LLM.

Outputs (all deterministic given the fixed seed):
  data/v0_5/tokenizer_corpus.txt   BPE training corpus (code + JSON + planner +
                                   a sample of the existing bilingual corpus)
  data/v0_5/train_corpus.txt       LM training corpus in the exact
                                   System / User / Assistant format used by
                                   LocalVoxlineProvider.chat()
  checkpoints/v0_5/planner_train.jsonl   structured training records (N>=300)
  checkpoints/v0_5/planner_val.jsonl     structured validation records (50)
  checkpoints/v0_5/planner_test.jsonl    structured held-out records (50)

The held-out (test) prompts are disjoint from train/validation: each record is
built from a unique (task_type, index) pair and an index-based split.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

SEED = 1337
ASSIST_START = "[ASSIST_START]"

# Planned counts (satisfy the >= 300 / 50 / 50 requirement).
N_TRAIN = 320
N_VAL = 50
N_TEST = 50

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "v0_5"
CKPT_DIR = REPO_ROOT / "checkpoints" / "v0_5"

# --------------------------------------------------------------------------
# Task template pool — each returns (user_request, plan_dict)
# --------------------------------------------------------------------------

MODULES = ["calculator", "utils", "parser", "fetcher", "serializer", "validator",
           "cache", "shipper", "loader", "formatter", "renderer", "scanner",
           "monitor", "pipeline", "adapter", "provider", "dispatcher", "scheduler",
           "indexer", "probe", "merger", "splitter", "compressor", "decoder",
           "encoder", "filter", "router", "halo", "beta_filter"]

WORDS = ["order", "user", "event", "batch", "report", "config", "session",
         "token", "payload", "metric", "log_line", "invoice", "profile",
         "record", "archive", "request", "response", "item", "entry", "field"]


def name_pair(gidx):
    """Deterministically derive a globally unique (file, prefix, function).

    The function name embeds the globally unique ``gidx`` so no two records (in
    any split) share an identical user request.
    """
    prefix = MODULES[gidx % len(MODULES)]
    base = WORDS[gidx % len(WORDS)]
    fn = f"{prefix}.py"
    func = f"{base}_{gidx}"
    return fn, prefix, func


def _plan_skeleton(objective, understanding, files, snippets, risks,
                   validation, requires_approval):
    steps = []
    for i, sn in enumerate(snippets):
        step = {
            "step_number": i + 1,
            "description": sn.get("description", ""),
            "action_type": sn.get("action_type", "read"),
            "target_files": sn.get("target_files", files),
            "command": sn.get("command", ""),
        }
        steps.append(step)
    return {
        "objective": objective,
        "understanding": understanding,
        "relevant_files": files,
        "steps": steps,
        "risks": risks,
        "validation_commands": validation,
        "requires_approval": requires_approval,
    }


def make_task(kind, idx):
    """Produce a (user_request, plan) pair deterministically from kind+idx.

    ``idx`` is a globally unique index across train/val/test, so the derived
    (file, function) pair is unique per record and no two records (within or
    across splits) share an identical user request.
    """
    fn, prefix, func = name_pair(idx)

    if kind == "create_file":
        req = f"Create a new module {fn} that provides a {func} function."
        plan = _plan_skeleton(
            objective=f"Create {fn} with a {func} utility",
            understanding="A new standalone module is needed; no existing file conflict.",
            files=[fn],
            snippets=[
                {"description": f"Create {fn}", "action_type": "write",
                 "target_files": [fn]},
                {"description": "Run a quick smoke import", "action_type": "command",
                 "target_files": [fn], "command": "python -m py_compile " + fn},
            ],
            risks=["Module may be larger than intended; keep scope minimal."],
            validation=[f"python -m py_compile {fn}"],
            requires_approval=False,
        )
    elif kind == "modify_function":
        loc = 40 + (idx % 60)
        req = f"Modify {func} in {fn} to handle empty input gracefully."
        plan = _plan_skeleton(
            objective=f"Update {func} in {fn}",
            understanding=f"Function {func} currently lacks a guard for empty input.",
            files=[fn],
            snippets=[
                {"description": f"Add empty-input guard in {func}",
                 "action_type": "write", "target_files": [fn]},
            ],
            risks=["Edge-case behavior may change; preserve existing return type."],
            validation=[f"python -c 'import {prefix}; print({col_call(fn, func)})'"],
            requires_approval=False,
        )
    elif kind == "add_test":
        req = f"Add unit tests for {func} in a test file for {prefix}."
        test_file = f"test_{prefix}.py"
        plan = _plan_skeleton(
            objective=f"Add unit tests for {func}",
            understanding="The module has no coverage for the main function.",
            files=[test_file],
            snippets=[
                {"description": f"Create {test_file}", "action_type": "write",
                 "target_files": [test_file]},
                {"description": "Run the test suite", "action_type": "command",
                 "target_files": [test_file], "command": "python -m pytest " + test_file},
            ],
            risks=["Tests may assume stable behavior; keep them appraisal-free."],
            validation=[f"python -m pytest {test_file}"],
            requires_approval=False,
        )
    elif kind == "fix_bug":
        line = 10 + (idx % 30)
        req = f"Fix the off-by-one bug in {func} inside {fn}."
        plan = _plan_skeleton(
            objective=f"Fix a correctness bug in {func}",
            understanding="A boundary condition produces incorrect results.",
            files=[fn],
            snippets=[
                {"description": f"Correct the boundary logic in {func}",
                 "action_type": "write", "target_files": [fn]},
            ],
            risks=["Fixing the boundary may expose other latent issues."],
            validation=[f"python -c 'import {prefix}; assert {col_call(fn, func)} is not None'"],
            requires_approval=False,
        )
    elif kind == "refactor":
        req = f"Refactor {func} in {fn} to use a helper and improve readability."
        plan = _plan_skeleton(
            objective=f"Refactor {func} for clarity",
            understanding="The function is long and mixes unrelated concerns.",
            files=[fn],
            snippets=[
                {"description": f"Extract a helper and simplify {func}",
                 "action_type": "write", "target_files": [fn]},
                {"description": "Run existing tests to confirm no regression",
                 "action_type": "command", "command": "python -m pytest"},
            ],
            risks=["Refactor may change behavior if not careful."],
            validation=["python -m pytest", f"python -m py_compile {fn}"],
            requires_approval=False,
        )
    elif kind == "add_module":
        pkg = prefix + "_pkg"
        req = f"Introduce a new package {pkg} and expose {func} through it."
        plan = _plan_skeleton(
            objective=f"Add package {pkg}",
            understanding="The codebase has no package for this concern.",
            files=[f"{pkg}/__init__.py", fn],
            snippets=[
                {"description": f"Create package skeleton {pkg}/__init__.py",
                 "action_type": "write", "target_files": [f"{pkg}/__init__.py"]},
                {"description": f"Add {func} to the package", "action_type": "write",
                 "target_files": [fn]},
            ],
            risks=["New package should not shadow an existing module."],
            validation=[f"python -c 'import {pkg}'"],
            requires_approval=False,
        )
    elif kind == "update_config":
        cfg = prefix + ".json"
        req = f"Update {cfg} to enable the new {func} feature flag."
        plan = _plan_skeleton(
            objective=f"Update configuration in {cfg}",
            understanding="A configuration file controls feature toggles.",
            files=[cfg],
            snippets=[
                {"description": f"Set feature flag in {cfg}", "action_type": "write",
                 "target_files": [cfg]},
                {"description": "Validate JSON syntax", "action_type": "command",
                 "command": "python -m json.tool " + cfg},
            ],
            risks=["Changing config may affect dependent modules."],
            validation=[f"python -m json.tool {cfg}"],
            requires_approval=False,
        )
    elif kind == "run_tests":
        req = (f"Run the test suite targeting {func} in module {prefix} "
               f"and report failing tests.")
        plan = _plan_skeleton(
            objective="Run the test suite and summarize",
            understanding="Tests are the primary confidence signal.",
            files=[],
            snippets=[
                {"description": "Run the test suite", "action_type": "command",
                 "target_files": [], "command": "python -m pytest"},
            ],
            risks=["Tests may take time or require dependencies."],
            validation=["python -m pytest"],
            requires_approval=False,
        )
    else:  # small_repo_change
        req = f"Make a small change to {prefix}: add a docstring and rename {func} to {func}_v2."
        plan = _plan_skeleton(
            objective=f"Apply a small change to {fn}",
            understanding="A tiny, low-risk change to documentation and naming.",
            files=[fn],
            snippets=[
                {"description": f"Rename {func}->{func}_v2 and add docstring",
                 "action_type": "write", "target_files": [fn]},
            ],
            risks=["Rename must be propagated to callers."],
            validation=[f"python -m py_compile {fn}"],
            requires_approval=False,
        )
    return req, plan


def col_call(fn, func):
    return func + "(x)"


# --------------------------------------------------------------------------
# Prompt formatting (mirrors CodingAgent._build_plan_prompt / LocalVoxline.chat)
# --------------------------------------------------------------------------

KINDS = ["create_file", "modify_function", "add_test", "fix_bug", "refactor",
         "add_module", "update_config", "run_tests", "small_repo_change"]


def format_user_prompt(req):
    return (
        "You are a software engineering assistant. Generate an implementation plan.\n"
        "User request: " + req + "\n"
        "Constraints: None\n"
        "Respond with ONLY a JSON object with keys objective, understanding, "
        "relevant_files, steps, risks, validation_commands, requires_approval."
    )


def format_line(req, plan):
    """Build the exact LocalVoxlineProvider.chat() format with the assist marker."""
    sys_prompt = "You are a helpful, correct coding assistant." + (
        "[LANG_ARMENIAN]" if 0 else ""
    )
    parts = [
        "System: " + sys_prompt,
        "User: " + format_user_prompt(req),
        "Assistant: " + ASSIST_START + " " + json.dumps(plan, ensure_ascii=False, sort_keys=False),
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Static auxiliary corpus (code + JSON) for the tokenizer / LM training
# --------------------------------------------------------------------------

PY_SNIPPETS = [
    "def add(a, b):\n    return a + b\n",
    "def parse_config(path):\n    import json\n    with open(path) as f:\n        return json.load(f)\n",
    "def fetch_data(url):\n    return requests.get(url).json()\n",
    "def validate_input(value):\n    if not value:\n        raise ValueError('missing value')\n    return value\n",
    "def compute_total(items):\n    return sum(item.get('price', 0) for item in items)\n",
    "def normalize(text):\n    return text.strip().lower()\n",
    "def format_output(d):\n    return ', '.join(f'{k}={v}' for k, v in d.items())\n",
    "def build_payload(user):\n    return {'name': user.name, 'email': user.email}\n",
    "import json\nCONFIG = {'debug': False, 'timeout': 30}\n",
    "def main():\n    result = compute_total(CONFIG)\n    print(result)\n\nif __name__ == '__main__':\n    main()\n",
]

JSON_SNIPPETS = [
    '{"objective": "Add a function", "understanding": "need a helper", "relevant_files": ["a.py"], "steps": [{"step_number": 1, "description": "write function", "action_type": "write", "target_files": ["a.py"], "command": ""}], "risks": [], "validation_commands": ["python -m py_compile a.py"], "requires_approval": false}',
    '{"name": "app", "version": "1.0.0", "dependencies": {"numpy": "*", "requests": "*"}}',
    '{"server": {"port": 8080, "host": "0.0.0.0"}, "workers": 4}',
    '{"steps": [{"action_type": "read", "target_files": ["b.py"]}], "requires_approval": false}',
]


def dcode_lines(n=400):
    """Deterministically generate a large, diverse pool of Python + JSON lines.

    Deliberately uses many distinct identifiers to broaden BPE merges so the
    tokenizer corpus can approach the target vocab size of 5000.
    """
    rng = random.Random(SEED + 77)
    verbs = ["build", "parse", "compute", "validate", "render", "load", "save",
             "merge", "split", "filter", "map_items", "reduce", "scan", "query",
             "compile", "deploy", "cache_hit", "normalize", "aggregate", "emit"]
    nouns = ["config", "schema", "payload", "response", "request", "model",
             "record", "entry", "bundle", "metric", "session", "tokenizer",
             "dataset", "pipeline", "registry", "store", "client", "server",
             "adapter", "handler"]
    lines = []
    for i in range(n):
        v = rng.choice(verbs)
        no = rng.choice(nouns)
        mod = rng.choice(MODULES)
        arg = rng.choice(["data", "items", "text", "batch", "params", "rows"])
        kind = i % 3
        if kind == 0:
            line = (f"def {v}_{no}_{i}({arg}):\n"
                    f"    result = 0\n"
                    f"    for x in {arg}:\n"
                    f"        result += x\n"
                    f"    return result\n")
        elif kind == 1:
            line = (f"def {v}_{no}_{i}({arg}, **opts):\n"
                    f"    from json import dumps\n"
                    f"    return dumps({{'module': '{mod}', 'value': {arg}, "
                    f"'opts': opts}})\n")
        else:
            line = (f"class {no.title()}{i}:\n"
                    f"    def __init__(self):\n"
                    f"        self.items = []\n"
                    f"    def {v}(self, {arg}):\n"
                    f"        self.items.append({arg})\n"
                    f"        return len(self.items)\n")
        lines.append(line)
    # diverse JSON configs
    for i in range(n // 2):
        cfg = {"module": rng.choice(MODULES) + str(i),
               "enabled": i % 2 == 0,
               "timeout": 10 + i,
               "endpoints": [f"/api/{v}/{no}" for v in [rng.choice(verbs)]],
               "nested": {"retries": i % 5, "pool": 2 + i % 8},
               "steps": [{"step_number": s, "action_type": "write",
                          "target_files": [f"{mod}_{i}.py"], "command": ""}
                         for s in range(1, 4)]}
        lines.append(json.dumps(cfg, ensure_ascii=False))
    return lines


def corpus_code_json_lines():
    lines = []
    for s in PY_SNIPPETS:
        lines.append(s)
    for s in JSON_SNIPPETS:
        lines.append(s)
    # large diverse pool to broaden BPE merges toward vocab 5000
    lines += dcode_lines(400)
    return lines


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def generate(args):
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    # Build planner records with disjoint (kind, idx) across splits.
    def records_for(n, start_mult):
        recs = []
        k = 0
        while len(recs) < n:
            kind = KINDS[k % len(KINDS)]
            idx = start_mult * n_test_base + k * 7 + 1  # deterministic spread
            req, plan = make_task(kind, idx)
            recs.append({"task_type": kind, "user_request": req, "plan": plan,
                         "prompt": format_user_prompt(req),
                         "line": format_line(req, plan)})
            k += 1
        return recs

    # ensure disjoint indices: use distinct base ranges
    n_test_base = args.n_val + args.n_test + 1
    train = records_for(args.n_train, 0)
    val = records_for(args.n_val, 1)
    test = records_for(args.n_test, 2)

    # write structured jsonl
    for name, recs in (("planner_train", train), ("planner_val", val),
                       ("planner_test", test)):
        with open(CKPT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # planner lines: train only feeds LM training; val fed to val_corpus; test
    # is held out entirely (evaluation only, never part of LM training).
    train_plan_lines = [r["line"] for r in train]
    val_plan_lines = [r["line"] for r in val]

    # tokenizer corpus: code + JSON + planner assistant bodies + general sample
    code_json = corpus_code_json_lines()
    general = load_general_sample(rng, args.general_size)
    tokenizer_lines = list(code_json)
    tokenizer_lines += [json.dumps(r["plan"], ensure_ascii=False)
                        for r in (train + val + test)]
    tokenizer_lines += general

    train_lm_lines = list(train_plan_lines)
    train_lm_lines += list(code_json)
    train_lm_lines += general

    with open(DATA_DIR / "tokenizer_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tokenizer_lines) + "\n")
    with open(DATA_DIR / "train_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_lm_lines) + "\n")
    with open(DATA_DIR / "val_corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_plan_lines) + "\n")

    return len(train), len(val), len(test), len(tokenizer_lines), len(train_lm_lines)


def load_general_sample(rng, size=800):
    """Deterministically sample a subset of the existing bilingual corpus."""
    corpus = REPO_ROOT / "data" / "voxline_corpus.txt"
    if not corpus.exists():
        return []
    with open(corpus, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    # seed on copy to keep planner generation reproducible
    srng = random.Random(SEED + 999)
    if len(lines) <= size:
        return lines
    return srng.sample(lines, size)


def main():
    ap = argparse.ArgumentParser(description="Build v0.5 planner dataset")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--general-size", type=int, default=800)
    args = ap.parse_args()

    nt, nv, nte, ntok, nlm = generate(args)
    print("Planner train:", nt)
    print("Planner val:", nv)
    print("Planner test(held-out):", nte)
    print("Tokenizer corpus lines:", ntok)
    print("Train LM corpus lines:", nlm)
    print("Outputs in:", DATA_DIR, "and", CKPT_DIR)


if __name__ == "__main__":
    main()
