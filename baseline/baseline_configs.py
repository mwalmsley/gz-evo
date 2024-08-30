from typing import Optional

from dataclasses import dataclass

# weak baseline
CFG_CONVNEXT_ATTO = dict(
    architecture_name="convnext_atto",
    v100_batch_size=512,
    a100_batch_size=2048,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXT_PICO = dict(
    architecture_name="convnext_pico",
    v100_batch_size=256,
    a100_batch_size=1024,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)


CFG_CONVNEXT_NANO = dict(
    architecture_name="convnext_nano",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXTV2_NANO_FCMAE = dict(
    architecture_name="convnextv2_nano.fcmae",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXTV2_NANO_FCMAE_FTIM = dict(
    architecture_name="convnextv2_nano.fcmae_ft_in22k_in1k",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)


CFG_CONVNEXT_BASE = dict(
    architecture_name="convnext_base",
    v100_batch_size=64,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXTV2_BASE_FCMAE= dict(
    architecture_name="convnextv2_base.fcmae",
    v100_batch_size=64,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXTV2_BASE_FCMAE_FTIM = dict(
    architecture_name="convnextv2_base.fcmae_ft_in22k_in1k",
    v100_batch_size=64,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)


CFG_RESNET50 = dict(
    architecture_name="resnet50",
    v100_batch_size=256,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05
)

# https://huggingface.co/timm/resnet50_clip.openai
CFG_RESNET50_CLIP = dict(
    architecture_name="resnet50_clip.openai",
    v100_batch_size=256,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05
)

CFG_EFFICIENTNET_B0 = dict(
    architecture_name="efficientnet_b0",
    v100_batch_size=256,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.2
)

CFG_EFFICIENTNETV2_S = dict(
    architecture_name="tf_efficientnetv2_s",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.2
)

CFG_MAXVIT_TINY = dict(
    architecture_name="maxvit_tiny_rw_224",
    v100_batch_size=64,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.2
)

# https://huggingface.co/timm?search_models=dinov2
# https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m
CFG_VIT_SMALL_DINO = dict(
    architecture_name="vit_small_patch14_reg4_dinov2.lvd142m",
    v100_batch_size=64,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-5,  # lower
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
    "convnext_atto": ModelConfig(**CFG_CONVNEXT_ATTO),
    "convnext_pico": ModelConfig(**CFG_CONVNEXT_PICO),
    "convnext_nano": ModelConfig(**CFG_CONVNEXT_NANO),
    "convnext_base": ModelConfig(**CFG_CONVNEXT_BASE),

    "convnextv2_nano.fcmae": ModelConfig(**CFG_CONVNEXTV2_NANO_FCMAE),
    "convnextv2_nano.fcmae_ft_in22k_in1k": ModelConfig(**CFG_CONVNEXTV2_NANO_FCMAE_FTIM),
    "convnextv2_base.fcmae": ModelConfig(**CFG_CONVNEXTV2_BASE_FCMAE),
    "convnextv2_base.fcmae_ft_in22k_in1k": ModelConfig(**CFG_CONVNEXTV2_BASE_FCMAE_FTIM),

    "resnet50": ModelConfig(**CFG_RESNET50),
    "resnet50_clip.openai": ModelConfig(**CFG_RESNET50_CLIP),
    
    "efficientnet_b0": ModelConfig(**CFG_EFFICIENTNET_B0),
    "tf_efficientnetv2_s": ModelConfig(**CFG_EFFICIENTNETV2_S),
    "maxvit_tiny_rw_224": ModelConfig(**CFG_MAXVIT_TINY),
    "vit_small_patch14_reg4_dinov2.lvd142m": ModelConfig(**CFG_VIT_SMALL_DINO)
}
