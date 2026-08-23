from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    vocab_size: int = 32000
    max_seq_length: int = 1024
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    save_steps: int = 500
    logging_steps: int = 50
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 42
    eval_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = False

    def resolve_paths(self) -> "TrainingConfig":
        self.checkpoint_dir = str(Path(self.checkpoint_dir).resolve())
        self.data_dir = str(Path(self.data_dir).resolve())
        return self


DEFAULT_CONFIG = TrainingConfig()
