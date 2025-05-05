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

CFG_CONVNEXT_NANO_FINETUNE = dict(
    architecture_name="convnext_nano",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4,
    n_blocks=-1, # still all blocks, but...
    lr_decay=0.7, # meaningful (aggressive) decay
    from_scratch=False
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

CFG_CONVNEXT_BASE_FINETUNE = dict(
    architecture_name="convnext_base",
    v100_batch_size=64,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4,
    n_blocks=-1, # still all blocks, but...
    lr_decay=0.7, # meaningful (aggressive) decay
    from_scratch=False
)

CFG_CONVNEXT_BASE_LAION = dict(
    architecture_name="convnext_base.clip_laion2b_augreg_ft_in12k",
    v100_batch_size=64,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXTV2_BASE_FCMAE= dict(
    architecture_name="convnextv2_base.fcmae",
    v100_batch_size=32,
    a100_batch_size=128,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXTV2_BASE_FCMAE_FTIM = dict(
    architecture_name="convnextv2_base.fcmae_ft_in22k_in1k",
    v100_batch_size=32,
    a100_batch_size=128,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)


CFG_RESNET50 = dict(
    architecture_name="resnet50",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05
)

# https://huggingface.co/timm/resnet50_clip.openai
CFG_RESNET50_CLIP = dict(
    architecture_name="resnet50_clip.openai",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5, 
    learning_rate=2e-5,  # reduced
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
# CFG_VIT_SMALL_DINO = dict(
#     architecture_name="vit_small_patch14_reg4_dinov2.lvd142m",
#     v100_batch_size=64,
#     a100_batch_size=256,
#     dropout_rate=0.5,
#     learning_rate=1e-5,  # lower
#     weight_decay=0.05
# )

CFG_VIT_BASE_CLIP = dict(
    architecture_name="vit_base_patch16_clip_224.openai",
    v100_batch_size=32,
    a100_batch_size=128,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05
)

CFG_VIT_MEDIUM_TINYCLIP = dict(
    architecture_name="vit_medium_patch32_clip_224.tinyclip_laion400m",
    v100_batch_size=32,
    a100_batch_size=128,
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
    n_blocks: Optional[int] = -1  # all blocks (no effect)
    lr_decay: Optional[float] = 1.  # no effect
    from_scratch: Optional[bool] = False  # no effect


MODEL_CONFIGS = {
    "convnext_atto": ModelConfig(**CFG_CONVNEXT_ATTO),
    "convnext_pico": ModelConfig(**CFG_CONVNEXT_PICO),
    "convnext_nano": ModelConfig(**CFG_CONVNEXT_NANO),
    "convnext_base": ModelConfig(**CFG_CONVNEXT_BASE),

    "convnext_nano_finetune": ModelConfig(**CFG_CONVNEXT_NANO_FINETUNE),
    "convnext_base_finetune": ModelConfig(**CFG_CONVNEXT_BASE_FINETUNE),

    "convnext_base.clip_laion2b_augreg_ft_in12k": ModelConfig(**CFG_CONVNEXT_BASE_LAION),

    "convnextv2_nano.fcmae": ModelConfig(**CFG_CONVNEXTV2_NANO_FCMAE),
    "convnextv2_nano.fcmae_ft_in22k_in1k": ModelConfig(**CFG_CONVNEXTV2_NANO_FCMAE_FTIM),
    "convnextv2_base.fcmae": ModelConfig(**CFG_CONVNEXTV2_BASE_FCMAE),
    "convnextv2_base.fcmae_ft_in22k_in1k": ModelConfig(**CFG_CONVNEXTV2_BASE_FCMAE_FTIM),

    "resnet50": ModelConfig(**CFG_RESNET50),
    "resnet50_clip.openai": ModelConfig(**CFG_RESNET50_CLIP),

    "efficientnet_b0": ModelConfig(**CFG_EFFICIENTNET_B0),
    "tf_efficientnetv2_s": ModelConfig(**CFG_EFFICIENTNETV2_S),
    "maxvit_tiny_rw_224": ModelConfig(**CFG_MAXVIT_TINY),
    # "vit_small_patch14_reg4_dinov2.lvd142m": ModelConfig(**CFG_VIT_SMALL_DINO)
    "vit_base_patch16_clip_224.openai": ModelConfig(**CFG_VIT_BASE_CLIP),
    "vit_medium_patch32_clip_224.tinyclip_laion400m": ModelConfig(**CFG_VIT_MEDIUM_TINYCLIP)


}


"""

Options

https://huggingface.co/models?search=timm/vit_base


timm/vit_base_patch16_clip_224.openai
timm/vit_base_patch32_clip_224.openai

21k only
timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
(no larger patch version)

timm/vit_base_patch16_clip_224.laion2b_ft_in12k
timm/vit_base_patch32_clip_224.laion2b_ft_in12k

added
timm/convnext_base.clip_laion2b_augreg_ft_in12k

timm/vit_medium_patch32_clip_224.tinyclip_laion400m

"""