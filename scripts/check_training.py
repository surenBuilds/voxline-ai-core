#!/usr/bin/env python3
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('checkpoints/v0_4/training_history.json', 'r', encoding='utf-8') as f:
    h = json.load(f)
print('Epochs completed:', len(h['train_loss']))
for i, (tl, vl) in enumerate(zip(h['train_loss'], h['val_loss']), 1):
    ppl = h.get('val_ppl', [0]*len(h['val_loss']))
    print(f'  Epoch {i}: train={tl:.4f} val={vl:.4f} ppl={ppl[i-1]:.1f}')
best = h.get('best_epoch', '?')
print(f'Best epoch: {best}')
print(f'Best val loss: {h.get("best_val_loss", "?")}')
