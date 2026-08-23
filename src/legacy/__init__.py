"""Voxline AI Legacy modules.

These modules are from v0.3 and are retained for reference only.
Do NOT import from this package in new code.

Canonical replacements:
  src/legacy/model.py     -> src/model/transformer.py (VoxlineTransformer)
  src/legacy/tokenizer.py -> src/tokenizer/bpe.py (BPETokenizer)
  src/legacy/config.py    -> src/training/trainer.py (TrainingConfig)
  src/legacy/dataset.py   -> src/training/trainer.py (LanguageModelDataset)
  src/legacy/train.py     -> src/training/trainer.py (Trainer)
  src/legacy/chat.py      -> src/api/chat.py (ConversationalAI)
"""

__all__ = []
