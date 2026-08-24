#!/usr/bin/env python3
"""Qwen Runtime Diagnostic Script.

Safely reports environment info and identifies the exact failure boundary
for loading and running the local Qwen2.5-0.5B-Instruct model.

Usage:
    python scripts/diagnose_qwen_runtime.py
"""

import sys
import os
import json
import time
from pathlib import Path

MODEL_PATH = Path("models/Qwen2.5-0.5B-Instruct")

results = {}


def stage(name):
    def decorator(fn):
        def wrapper():
            try:
                start = time.time()
                result = fn()
                elapsed = (time.time() - start) * 1000
                results[name] = {"status": "OK", "detail": result, "ms": round(elapsed)}
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                results[name] = {"status": "FAIL", "error": str(e), "type": type(e).__name__, "ms": round(elapsed)}
                return None
        wrapper._stage_name = name
        return wrapper
    return decorator


@stage("A_python")
def check_python():
    return sys.version


@stage("B_os")
def check_os():
    import platform
    return f"{platform.system()} {platform.release()} ({platform.version()})"


@stage("C_cpu")
def check_cpu():
    import platform
    return platform.processor() or "unknown"


@stage("D_ram")
def check_ram():
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        total_gb = mem.ullTotalPhys / (1024 ** 3)
        avail_gb = mem.ullAvailPhys / (1024 ** 3)
        return f"{total_gb:.1f} GB total, {avail_gb:.1f} GB available"
    except Exception:
        return "unable to detect"


@stage("E_torch")
def check_torch():
    import torch
    return f"{torch.__version__} (cuda={torch.cuda.is_available()}, cpu=True)"


@stage("F_transformers")
def check_transformers():
    import transformers
    return transformers.__version__


@stage("G_safetensors")
def check_safetensors():
    import safetensors
    return safetensors.__version__


@stage("H_tokenizers")
def check_tokenizers():
    import tokenizers
    return tokenizers.__version__


@stage("I_numpy")
def check_numpy():
    import numpy
    return numpy.__version__


@stage("J_model_files")
def check_model_files():
    if not MODEL_PATH.is_dir():
        return f"Model directory not found: {MODEL_PATH}"
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
    weight_files = list(MODEL_PATH.glob("*.safetensors"))
    missing = [f for f in required if not (MODEL_PATH / f).exists()]
    sizes = {}
    for f in list(MODEL_PATH.glob("*")):
        if f.is_file():
            sizes[f.name] = f"{f.stat().st_size / 1024 / 1024:.1f} MB"
    return {
        "missing_required": missing,
        "weight_files": [f.name for f in weight_files],
        "weight_sizes": {f.name: f"{f.stat().st_size / 1024 / 1024:.1f} MB" for f in weight_files},
        "all_files": sizes,
    }


@stage("K_config")
def check_config():
    config_path = MODEL_PATH / "config.json"
    if not config_path.exists():
        return "config.json not found"
    with open(config_path) as f:
        cfg = json.load(f)
    return {
        "model_type": cfg.get("model_type"),
        "vocab_size": cfg.get("vocab_size"),
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
    }


@stage("L_tokenizer_load")
def check_tokenizer():
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    vocab_size = tok.vocab_size
    has_chat_template = hasattr(tok, "chat_template") and tok.chat_template is not None
    test_tokens = tok("Hello, world!", return_tensors="pt")
    return {
        "vocab_size": vocab_size,
        "has_chat_template": has_chat_template,
        "test_encoding_shape": list(test_tokens.input_ids.shape),
    }


@stage("M_model_load")
def check_model_load():
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        dtype=torch.float32,
    )
    param_count = sum(p.numel() for p in model.parameters())
    model.eval()
    return {
        "parameters": param_count,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }


@stage("N_forward_pass")
def check_forward_pass():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, dtype=torch.float32,
    )
    model.eval()
    inputs = tok("Hello", return_tensors="pt")
    with torch.inference_mode():
        output = model(**inputs)
    return {"logits_shape": list(output.logits.shape)}


@stage("O_generate")
def check_generate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, dtype=torch.float32,
    )
    model.eval()
    inputs = tok("Hello", return_tensors="pt")
    with torch.inference_mode():
        output_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return {"generated_text": text, "output_length": len(output_ids[0])}


@stage("P_chat_template_generate")
def check_chat_template_generate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, dtype=torch.float32,
    )
    model.eval()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]
    chat_input = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    with torch.inference_mode():
        output_ids = model.generate(
            chat_input.input_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new_tokens = output_ids[0, chat_input.input_ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return {"generated_text": text}


def main():
    print("=" * 60)
    print("VOXLINE QWEN RUNTIME DIAGNOSTIC")
    print("=" * 60)
    print()

    stages = [
        s for s in [
            check_python, check_os, check_cpu, check_ram,
            check_torch, check_transformers, check_safetensors, check_tokenizers, check_numpy,
            check_model_files, check_config, check_tokenizer,
            check_model_load, check_forward_pass, check_generate, check_chat_template_generate,
        ]
    ]

    first_failure = None
    for s in stages:
        name = getattr(s, "_stage_name", s.__name__)
        print(f"  {name} ... ", end="", flush=True)
        result = s()
        info = results[name]
        if info["status"] == "OK":
            print(f"OK ({info['ms']}ms)")
        else:
            print(f"FAIL: {info['type']}: {info['error'][:100]}")
            if first_failure is None:
                first_failure = name

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok_count = sum(1 for v in results.values() if v["status"] == "OK")
    fail_count = sum(1 for v in results.values() if v["status"] == "FAIL")
    print(f"  Passed: {ok_count}/{ok_count + fail_count}")
    if first_failure:
        print(f"  First failure: {first_failure}")
        info = results[first_failure]
        print(f"  Error: {info['type']}: {info['error']}")
    else:
        print("  All stages passed!")
    print()

    output_path = Path("eval_results") / "qwen_diagnostic.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()
