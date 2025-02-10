from dataclasses import dataclass
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Callable


@dataclass
class Args:
    model_name: str = None
    dataset_name: str = None
    device: int = None
    output_dir: str = None
    batch_size: int = None
    gradient_accumulation_steps: int = None
    lr: float = None
    l2reg: float = None
    num_epochs: int = None
    optimizer_name: str = None
    scheduler_type: str = None
    initializer_name: str = None
    warmup_ratio: float = None
    logging_steps: int = None
    seed: int = None
    early_stopping_patience: int = None
    num_classes: int = None
    dropout: float = None

    model_class: nn.Module = None
    initializer: Callable = None
    criterion: CrossEntropyLoss = None
    pretrained_path: str = None
