#!/usr/bin/env python3
"""Performance audit script."""
import sys; sys.stdout.reconfigure(encoding='utf-8'); sys.path.insert(0, '.')
import time, torch, json, tracemalloc

from src.tokenizer.bpe import BPETokenizer
from src.model.transformer import VoxlineTransformer
from src.inference.generator import TextGenerator, GenerationConfig

# Measure load time
t0 = time.time()
tok = BPETokenizer(vocab_size=5000)
tok.load('checkpoints/v0_4/tokenizer.json')
ckpt = torch.load('checkpoints/v0_4/best_model.pt', map_location='cpu', weights_only=False)
model = VoxlineTransformer(vocab_size=tok.get_vocab_size(), d_model=128, num_layers=4, num_heads=4, d_ff=512, max_seq_len=128, dropout=0.1)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
load_time = time.time() - t0

# Measure RAM
tracemalloc.start()

gen = TextGenerator(model, tok, device='cpu')
config = GenerationConfig(max_new_tokens=50, temperature=0.7, top_p=0.9, do_sample=True, eos_token_id=tok.vocab.get('<EOS>'))

# Benchmark inference
prompts = ['Hello', 'What is AI', 'Barev', 'Science', 'Machine learning']
times = []
token_counts = []
for p in prompts:
    t = time.time()
    r = gen.generate(p, config)
    elapsed = time.time() - t
    times.append(elapsed)
    # Count output tokens (approximate)
    token_counts.append(len(tok.encode(r)) if r else 0)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f'=== PERFORMANCE AUDIT ===')
print(f'Model load time: {load_time:.2f}s')
print(f'Inference times per prompt (50 tokens):')
for p, t in zip(prompts, times):
    print(f'  "{p}": {t:.2f}s')
print(f'Avg inference time: {sum(times)/len(times):.2f}s')
print(f'Est. tokens/sec: {50/(sum(times)/len(times)):.1f}')
print(f'RAM current: {current/1024/1024:.1f} MB')
print(f'RAM peak: {peak/1024/1024:.1f} MB')
print(f'Checkpoint size: 11.11 MB')
print(f'Has optimizer state: {bool(ckpt.get("optimizer_state_dict"))}')
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# Test training history
hist = ckpt.get('training_history', [])
if hist:
    print(f'\n=== TRAINING HISTORY ===')
    print(f'Epochs completed: {len(hist)}')
    print(f'Best val_loss: {min(h["val_loss"] for h in hist):.4f}')
    print(f'Best perplexity: {min(h["perplexity"] for h in hist):.2f}')
    print(f'Final train_loss: {hist[-1]["train_loss"]:.4f}')
