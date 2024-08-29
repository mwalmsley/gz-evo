from typing import Optional

from dataclasses import dataclass


CFG_CONVNEXT_NANO = dict(
    architecture_name="convnext_nano",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)


CFG_RESNET50 = dict(
    architecture_name="resnet50",
    v100_batch_size=512,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05
)

# https://huggingface.co/timm/resnet50_clip.openai
CFG_RESNET50_CLIP = dict(
    architecture_name="resnet50_clip.openai",
    v100_batch_size=512,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05
)

@dataclass
class ModelConfig:
    architecture_name: str
    v100_batch_size: int
    a100_batch_size: int
    dropout_rate: float
    learning_rate: float
    weight_decay: float
    drop_path_rate: Optional[float] = 0.

MODEL_CONFIGS = {
    "convnext_nano": ModelConfig(**CFG_CONVNEXT_NANO),
    "resnet50": ModelConfig(**CFG_RESNET50),
    "resnet50_clip": ModelConfig(**CFG_RESNET50_CLIP),
}



