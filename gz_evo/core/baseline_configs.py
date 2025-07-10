from typing import Optional

from dataclasses import dataclass

@dataclass
class ModelConfig:
    architecture_name: str
    v100_batch_size: int
    a100_batch_size: int
    dropout_rate: float
    learning_rate: float
    weight_decay: float
    drop_path_rate: Optional[float] = 0.
    layer_decay: Optional[float] = 1.  # no effect


# weak baseline
CFG_CONVNEXT_ATTO = ModelConfig(
    architecture_name="convnext_atto",
    v100_batch_size=512,
    a100_batch_size=2048,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
    
)

CFG_CONVNEXT_PICO = ModelConfig(
    architecture_name="convnext_pico",
    v100_batch_size=256,
    a100_batch_size=1024,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)


CFG_CONVNEXT_NANO = ModelConfig(
    architecture_name="convnext_nano",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXT_NANO_FINETUNE = ModelConfig(
    architecture_name="convnext_nano",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4,
    layer_decay=0.7, # meaningful (aggressive) decay
)


CFG_CONVNEXTV2_NANO_FCMAE = ModelConfig(
    architecture_name="convnextv2_nano.fcmae",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXTV2_NANO_FCMAE_FTIM = ModelConfig(
    architecture_name="convnextv2_nano.fcmae_ft_in22k_in1k",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)


CFG_CONVNEXT_BASE = ModelConfig(
    architecture_name="convnext_base",
    v100_batch_size=64,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXT_BASE_FINETUNE = ModelConfig(
    architecture_name="convnext_base",
    v100_batch_size=64,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4,
    layer_decay=0.7 # meaningful (aggressive) decay
)

CFG_CONVNEXT_BASE_LAION = ModelConfig(
    architecture_name="convnext_base.clip_laion2b_augreg_ft_in12k",
    v100_batch_size=64,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXT_LARGE = ModelConfig(
    architecture_name="convnext_large",
    v100_batch_size=32,
    a100_batch_size=128,
    dropout_rate=0.5,
    learning_rate=2e-5,  # reduced
    weight_decay=0.05,
    drop_path_rate=0.4
)


CFG_CONVNEXTV2_BASE_FCMAE= ModelConfig(
    architecture_name="convnextv2_base.fcmae",
    v100_batch_size=32,
    a100_batch_size=128,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)

CFG_CONVNEXTV2_BASE_FCMAE_FTIM = ModelConfig(
    architecture_name="convnextv2_base.fcmae_ft_in22k_in1k",
    v100_batch_size=32,
    a100_batch_size=128,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.4
)


CFG_RESNET50 = ModelConfig(
    architecture_name="resnet50",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05
)

# https://huggingface.co/timm/resnet50_clip.openai
CFG_RESNET50_CLIP = ModelConfig(
    architecture_name="resnet50_clip.openai",
    v100_batch_size=128,
    a100_batch_size=512,
    dropout_rate=0.5, 
    learning_rate=2e-5,  # reduced
    weight_decay=0.05
)

CFG_EFFICIENTNET_B0 = ModelConfig(
    architecture_name="efficientnet_b0",
    v100_batch_size=256,
    a100_batch_size=512,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.2
)


# possibly effnet v2 needs
# LEARNING_RATE=0.0001
# DROP_PATH_RATE=0.0
# WEIGHT_DECAY=0.0


CFG_EFFICIENTNETV2_S = ModelConfig(
    architecture_name="tf_efficientnetv2_s",
    v100_batch_size=128,
    a100_batch_size=256,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.2
)

CFG_EFFICIENTNETV2_M = ModelConfig(
    architecture_name="tf_efficientnetv2_m",
    v100_batch_size=64,
    a100_batch_size=128,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.2
)

CFG_EFFICIENTNETV2_L = ModelConfig(
    architecture_name="tf_efficientnetv2_l",
    v100_batch_size=32,
    a100_batch_size=64,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.2
)

CFG_MAXVIT_TINY = ModelConfig(
    architecture_name="maxvit_tiny_rw_224",
    v100_batch_size=64,
    a100_batch_size=256,  
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.2
)

CFG_MAXVIT_SMALL = ModelConfig(
    architecture_name="maxvit_rmlp_small_rw_224",
    v100_batch_size=32,
    a100_batch_size=128,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05,
    drop_path_rate=0.3
)

CFG_MAXVIT_BASE = ModelConfig(
    architecture_name="maxvit_rmlp_base_rw_224",
    v100_batch_size=16,
    a100_batch_size=64,
    dropout_rate=0.5,
    learning_rate=5e-5,
    weight_decay=0.05,
    drop_path_rate=0.45  # guess
)


CFG_MAXVIT_LARGE = ModelConfig(
    architecture_name="maxvit_large_tf_224",
    v100_batch_size=8,
    a100_batch_size=32,
    dropout_rate=0.5,
    learning_rate=2e-5,
    weight_decay=0.05,
    drop_path_rate=0.6
)


# https://huggingface.co/timm?search_models=dinov2
# https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m
# CFG_VIT_SMALL_DINO = ModelConfig(
#     architecture_name="vit_small_patch14_reg4_dinov2.lvd142m",
#     v100_batch_size=64,
#     a100_batch_size=256,
#     dropout_rate=0.5,
#     learning_rate=1e-5,  # lower
#     weight_decay=0.05
# )

CFG_VIT_BASE_CLIP = ModelConfig(
    architecture_name="vit_base_patch16_clip_224.openai",
    v100_batch_size=32,
    a100_batch_size=128,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05
)

CFG_VIT_MEDIUM_TINYCLIP = ModelConfig(
    architecture_name="vit_medium_patch32_clip_224.tinyclip_laion400m",
    v100_batch_size=32,
    a100_batch_size=128,
    dropout_rate=0.5,
    learning_rate=1e-4,
    weight_decay=0.05
)

CFG_VIT_SO400M_SIGLIP = ModelConfig(
    architecture_name="vit_so400m_patch14_siglip_224.v2_webli",
    v100_batch_size=8,  # will likely fail, need higher cuda compute capability
    a100_batch_size=64,  # trying it
    dropout_rate=0.5,
    learning_rate=1e-5,  # may be low
    weight_decay=0.05
)

CFG_VIT_SO400M_SIGLIP_FINETUNE = ModelConfig(
    architecture_name="vit_so400m_patch14_siglip_224.v2_webli",
    v100_batch_size=8,  # will likely fail, need higher cuda compute capability
    a100_batch_size=64,  # trying it
    dropout_rate=0.5,
    learning_rate=1e-5,  # may be low
    weight_decay=0.05,
    layer_decay=0.5
)




MODEL_CONFIGS = {
    "convnext_atto": CFG_CONVNEXT_ATTO,
    "convnext_pico": CFG_CONVNEXT_PICO,
    "convnext_nano": CFG_CONVNEXT_NANO,
    "convnext_base": CFG_CONVNEXT_BASE,
    "convnext_large": CFG_CONVNEXT_LARGE,

    "convnext_nano_finetune": CFG_CONVNEXT_NANO_FINETUNE,
    "convnext_base_finetune": CFG_CONVNEXT_BASE_FINETUNE,

    "convnext_base.clip_laion2b_augreg_ft_in12k": CFG_CONVNEXT_BASE_LAION,

    "convnextv2_nano.fcmae": CFG_CONVNEXTV2_NANO_FCMAE,
    "convnextv2_nano.fcmae_ft_in22k_in1k": CFG_CONVNEXTV2_NANO_FCMAE_FTIM,
    "convnextv2_base.fcmae": CFG_CONVNEXTV2_BASE_FCMAE,
    "convnextv2_base.fcmae_ft_in22k_in1k": CFG_CONVNEXTV2_BASE_FCMAE_FTIM,

    "resnet50": CFG_RESNET50,
    "resnet50_clip.openai": CFG_RESNET50_CLIP,

    "efficientnet_b0": CFG_EFFICIENTNET_B0,

    "tf_efficientnetv2_s": CFG_EFFICIENTNETV2_S,
    "tf_efficientnetv2_m": CFG_EFFICIENTNETV2_M,
    "tf_efficientnetv2_l": CFG_EFFICIENTNETV2_L,

    "maxvit_tiny": CFG_MAXVIT_TINY,
    "maxvit_small": CFG_MAXVIT_SMALL,
    "maxvit_base": CFG_MAXVIT_BASE,
    "maxvit_large": CFG_MAXVIT_LARGE,

    # "vit_small_patch14_reg4_dinov2.lvd142m": CFG_VIT_SMALL_DINO)
    "vit_base_patch16_clip_224.openai": CFG_VIT_BASE_CLIP,
    "vit_medium_patch32_clip_224.tinyclip_laion400m": CFG_VIT_MEDIUM_TINYCLIP,

    "vit_so400m_siglip": CFG_VIT_SO400M_SIGLIP,  # same but no layer_decay
    "vit_so400m_siglip_finetune": CFG_VIT_SO400M_SIGLIP_FINETUNE

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