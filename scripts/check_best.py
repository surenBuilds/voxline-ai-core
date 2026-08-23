#!/usr/bin/env python3
import sys, torch, json, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.transformer import VoxlineTransformer
from src.tokenizer.bpe import BPETokenizer

tok = BPETokenizer(vocab_size=5000)
tok.load('checkpoints/v0_4/tokenizer.json')
print(f'Tokenizer vocab: {tok.get_vocab_size()}')

model = VoxlineTransformer(
    vocab_size=tok.get_vocab_size(),
    d_model=128, num_layers=4, num_heads=4, d_ff=512,
    max_seq_len=128, dropout=0.1
)
ckpt = torch.load('checkpoints/v0_4/best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f'Best epoch: {ckpt.get("epoch", "?")}')
print(f'Best val loss: {ckpt.get("val_loss", "?")}')

history_path = 'checkpoints/v0_4/training_history.json'
if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        h = json.load(f)
    for i, (tl, vl) in enumerate(zip(h['train_loss'], h['val_loss']), 1):
        ppl = h.get('val_ppl', [0]*len(h['val_loss']))
        print(f'  Epoch {i}: train={tl:.4f} val={vl:.4f} ppl={ppl[i-1]:.1f}')
