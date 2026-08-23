#!/usr/bin/env python3
import sys, torch, json, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

ckpt = torch.load('checkpoints/v0_4/best_model.pt', map_location='cpu', weights_only=False)
print("Keys:", list(ckpt.keys()))
for k in ckpt:
    if k not in ('model_state_dict', 'optimizer_state_dict'):
        print(f"  {k}: {ckpt[k]}")
print(f"  global_step: {ckpt.get('global_step', 'N/A')}")
if 'training_history' in ckpt:
    for entry in ckpt['training_history']:
        print(f"  Epoch {entry['epoch']}: train_loss={entry['train_loss']:.4f} val_loss={entry['val_loss']:.4f} ppl={entry['perplexity']:.1f}")

from src.tokenizer.bpe import BPETokenizer
from src.model.transformer import VoxlineTransformer

tok = BPETokenizer(vocab_size=5000)
tok.load('checkpoints/v0_4/tokenizer.json')
print(f"\nTokenizer vocab: {tok.get_vocab_size()}")

model = VoxlineTransformer(
    vocab_size=tok.get_vocab_size(),
    d_model=128, num_layers=4, num_heads=4, d_ff=512,
    max_seq_len=128, dropout=0.1
)
model.load_state_dict(ckpt['model_state_dict'])
num_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {num_params:,}")

from src.inference.generator import TextGenerator, GenerationConfig
gen = TextGenerator(model, tok, device="cpu")
prompts = ["Hello, how are you?", "What is AI?", "Բարև", "Ինչպե՞ս ես", "Machine learning is", "Տեխնոլոգիան"]
config = GenerationConfig(
    max_new_tokens=50, temperature=0.7, top_k=40, top_p=0.9,
    eos_token_id=tok.vocab.get("<EOS>"), pad_token_id=tok.vocab.get("<PAD>"),
    do_sample=True,
)
for p in prompts:
    out = gen.generate(p, config)
    print(f"  {p} -> {out}")
