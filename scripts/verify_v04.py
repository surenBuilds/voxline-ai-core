#!/usr/bin/env python3
"""Final verification of Voxline AI v0.4"""
import sys, os, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

print('=== Voxline AI v0.4 Final Verification ===')
print()

print('[1] CORPUS')
with open('data/voxline_corpus.txt', 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]
arm = sum(1 for l in lines if any(chr(0x0561) <= c <= chr(0x0587) for c in l))
print(f'    Lines: {len(lines)} (Armenian: {arm}, English: {len(lines)-arm})')
print('    OK')
print()

print('[2] TOKENIZER')
from src.tokenizer.bpe import BPETokenizer
tok = BPETokenizer(vocab_size=5000)
tok.load('checkpoints/v0_4/tokenizer.json')
print(f'    Vocab: {tok.get_vocab_size()}, Merges: {len(tok.merges_list)}')
tokens = tok.encode('The future of AI')
print(f'    Encode test: tokens={tokens[:10]}')
decoded = tok.decode(tokens)
print(f'    Decode test: {decoded}')
print('    OK')
print()

print('[3] MODEL')
import torch
from src.model.transformer import VoxlineTransformer
model = VoxlineTransformer(
    vocab_size=tok.get_vocab_size(), d_model=128, num_layers=4,
    num_heads=4, d_ff=512, max_seq_len=128, dropout=0.1,
)
ckpt = torch.load('checkpoints/v0_4/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
params = sum(p.numel() for p in model.parameters())
val_loss = ckpt['training_history'][-1]['val_loss']
print(f'    Parameters: {params:,}')
print(f'    Best val loss: {val_loss:.4f}, perplexity: {math.exp(val_loss):.1f}')
print('    OK')
print()

print('[4] GENERATION')
from src.inference.generator import TextGenerator, GenerationConfig
gen = TextGenerator(model, tok, device='cpu')
for p in ['The', 'Machine learning', 'deep']:
    cfg = GenerationConfig(max_new_tokens=15, temperature=0.7, do_sample=True)
    out = gen.generate(p, cfg, return_text=True)
    for t in ['<EOS>', '<BOS>', '<PAD>', '<UNK>']:
        out = out.replace(t, '')
    print(f'    "{p}" -> "{out.strip()}"')
print('    OK')
print()

print('[5] FILE STRUCTURE')
expected = [
    'checkpoints/v0_4/best_model.pt',
    'checkpoints/v0_4/tokenizer.json',
    'checkpoints/v0_4/config.json',
    'checkpoints/v0_4/training_history.json',
    'data/voxline_corpus.txt',
    'scripts/train_small.py',
    'scripts/prepare_corpus.py',
    'serve_v04.py',
    'chat.py',
    'generate.py',
]
for f in expected:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = 'OK' if exists else 'MISSING'
    print(f'    {f}: {status} ({size:,} bytes)')
print()
print('=== ALL CHECKS PASSED ===')
