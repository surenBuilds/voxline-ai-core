"""
VOXLINE AI CORE v0.3 - VERIFICATION CHECKPOINT RELOAD AND INFERENCE
Verify that saved checkpoints can be reloaded and perform inference correctly.
"""

import sys
from pathlib import Path

import torch
from torch import nn

from src.tokenizer import SimpleTokenizer

# ============================================================
# NextTokenModel (same as in main.py)
# ============================================================
class NextTokenModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, seq_length=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(seq_length, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, vocab_size),
        )

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        x = self.embedding(input_ids) + self.position(positions)
        pooled = x.mean(dim=1)
        return self.net(pooled)


def verify_checkpoint_reload():
    """Verify that v0.3 checkpoints can be loaded and used for inference."""
    
    print("\n" + "=" * 70)
    print("VOXLINE AI CORE v0.3 - CHECKPOINT VERIFICATION")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Step 1: Load tokenizer
    print("\n[1/3] Loading tokenizer checkpoint...")
    try:
        tokenizer = SimpleTokenizer()
        tokenizer.load("checkpoints/voxline_tokenizer_v0_3.json")
        print(f"✓ Tokenizer loaded")
        print(f"  Vocabulary size: {len(tokenizer.vocab)}")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False
    
    # Step 2: Load model checkpoint
    print("\n[2/3] Loading model checkpoint...")
    try:
        model = NextTokenModel(vocab_size=len(tokenizer.vocab), seq_length=8).to(device)
        model.load_state_dict(torch.load("checkpoints/voxline_ai_v0_3.pt", map_location=device))
        model.eval()
        print(f"✓ Model loaded")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Step 3: Perform inference
    print("\n[3/3] Testing inference with loaded checkpoint...")
    try:
        test_prompts = [
            "Voxline creates",
            "The tokenizer",
            "AI learns",
        ]
        
        for prompt in test_prompts:
            input_ids = tokenizer.encode(prompt)
            if len(input_ids) > 8:
                input_ids = input_ids[-8:]
            else:
                input_ids = [tokenizer.vocab["<PAD>"]] * (8 - len(input_ids)) + input_ids
            
            with torch.no_grad():
                logits = model(torch.tensor([input_ids], dtype=torch.long).to(device))
                top_idx = torch.argmax(logits, dim=1).item()
                predicted_token = tokenizer.inv_vocab.get(top_idx, "<UNK>")
            
            print(f"  Prompt: '{prompt}' → Predicted: '{predicted_token}'")
        
        print(f"✓ Inference successful")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("CHECKPOINT RELOAD: PASS ✓")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    success = verify_checkpoint_reload()
    sys.exit(0 if success else 1)
