#!/usr/bin/env python3
"""Quick generation test for v0.4 model."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.model.transformer import VoxlineTransformer
from src.tokenizer.bpe import BPETokenizer
import yaml

with open('configs/model_configs.yaml') as f:
    m = yaml.safe_load(f)['v0_4_small']['model']

tok = BPETokenizer(vocab_size=m['vocab_size'])
tok.load('checkpoints/v0_4/tokenizer.json')

model = VoxlineTransformer(
    vocab_size=tok.get_vocab_size(),
    d_model=m['d_model'], num_layers=m['num_layers'],
    num_heads=m['num_heads'], d_ff=m['d_ff'],
    max_seq_len=m['max_seq_len'], dropout=m['dropout'],
)
ckpt = torch.load('checkpoints/v0_4/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

def generate(prompt, max_new_tokens=30, temperature=0.8):
    ids = tok.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        input_crop = input_ids[:, -m['max_seq_len']:]
        with torch.no_grad():
            logits = model(input_crop)
        next_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id.item() == tok.token_to_id('<EOS>'):
            break
    return tok.decode(input_ids[0].tolist())

print('--- v0.4 Generation Tests ---')
for p in ['The', 'Machine learning', 'Artificial', 'deep', 'transformer']:
    out = generate(p)
    print(f'  "{p}" -> "{out}"')
