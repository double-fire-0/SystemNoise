from .mobilenet_v2 import mobilenet_v2  # noqa: F401
from .regnet import (  # noqa: F401
    regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
    regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
    regnety_200m, regnety_400m, regnety_600m, regnety_800m,
    regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m,
)

from .efficientnet import (  # noqa: F401
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    efficientnet_b0_nodrop, efficientnet_b1_nodrop, efficientnet_b2_nodrop, efficientnet_b3_nodrop,
    efficientnet_b4_nodrop, efficientnet_b5_nodrop, efficientnet_b6_nodrop, efficientnet_b7_nodrop
)
from .shufflenet_v2 import (  # noqa: F401
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0, shufflenet_v2_scale
)

from .densenet import densenet121, densenet169, densenet201, densenet161  # noqa: F401
# from .toponet import toponet_conv, toponet_sepconv, toponet_mb

from .resnet_official import (  # noqa: F401
    resnet18_official, resnet34_official, resnet50_official, resnet101_official, resnet152_official,
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
)

from .mobilenet_v3 import mobilenet_v3  # noqa: F401

from .vision_transformer import (  # noqa: F401
    vit_b32_224, vit_b16_224,
    deit_tiny_b16_224, deit_small_b16_224, deit_base_b16_224
)

# add from .vit
from .vit.swin_transformer import (  # noqa: F401
    swin_tiny, swin_small, swin_base_224, swin_base_384, swin_large_224, swin_large_384
)
from .vit.mlp_mixer import (  # noqa: F401
    mixer_b16_224, mixer_L16_224
)
from .vit.vit_base import (  # noqa: F401
    new_deit_small_patch16_224
)
from .repvgg import (  # noqa: F401
    repvgg_A0, repvgg_A1, repvgg_A2, repvgg_B0,
    repvgg_B1, repvgg_B1g2, repvgg_B1g4,
    repvgg_B2, repvgg_B2g2, repvgg_B2g4,
    repvgg_B3, repvgg_B3g2, repvgg_B3g4
)

from .alexnet import alexnet



def get_model_robust_dcit():
    return {
        'alexnet': alexnet(),
        "deit_base_b16_224": deit_base_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True),
        "deit_small_b16_224": deit_small_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True),
        "deit_tiny_b16_224": deit_tiny_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True),
        "densenet121": densenet121(),
        "densenet169": densenet169(),
        "densenet201": densenet201(),
        "mixer_b16_224": mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        "mixer_L16_224": mixer_L16_224(drop_path=0.0, drop_path_rate=0.0),
        "mobilenet_v2_x0_5": mobilenet_v2(scale=0.5),
        "mobilenet_v2_x0_75": mobilenet_v2(scale=0.75),
        "mobilenet_v2_x1_0": mobilenet_v2(scale=1.0),
        "mobilenet_v2_x1_4": mobilenet_v2(scale=1.4),
        "mobilenet_v3_large_x0_5": mobilenet_v3(scale=0.5, dropout=0.0, mode='large'),
        "mobilenet_v3_large_x0_35": mobilenet_v3(scale=0.35, dropout=0.0, mode='large'),
        "mobilenet_v3_large_x0_75": mobilenet_v3(scale=0.75, dropout=0.0, mode='large'),
        "mobilenet_v3_large_x1_0": mobilenet_v3(scale=1.0, dropout=0.0, mode='large'),
        "mobilenet_v3_large_x1_4": mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        "regnetx_400m": regnetx_400m(),
        "regnetx_800m": regnetx_800m(),
        "regnetx_1600m": regnetx_1600m(),
        "regnetx_3200m": regnetx_3200m(),
        "regnetx_6400m": regnetx_6400m(),
        "repvgg_A0": repvgg_A0(),
        "repvgg_B3": repvgg_B3(),
        "resnet18": resnet18_official(),
        "resnet34": resnet34_official(),
        "resnet50": resnet50_official(),
        "resnet101": resnet101_official(),
        "resnet152": resnet152_official(),
        "resnext50_32x4d": resnext50_32x4d(),
        "resnext101_32x8d": resnext101_32x8d(),
        "shufflenet_v2_x0_5": shufflenet_v2_x0_5(),
        "shufflenet_v2_x1_0": shufflenet_v2_x1_0(),
        "shufflenet_v2_x1_5": shufflenet_v2_x1_5(),
        "shufflenet_v2_x2_0": shufflenet_v2_x2_0(),
        "vit_b16_224": vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                   representation_size=768),
        "vit_b32_224": vit_b32_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                   representation_size=768),
        "wide_resnet50_2": wide_resnet50_2(),
        "wide_resnet101_2": wide_resnet101_2(),
        'resnet50_augmix': resnet50_official(),
        'resnet50_mococv2': resnet50_official(),
        'resnet50_adamw': resnet50_official(),
        'regnetx_3200m_augmix': regnetx_3200m(),
        'regnetx_3200m_adamw': regnetx_3200m(),
        'repvgg_A0_deploy': repvgg_A0(),
        'repvgg_B3_deploy': repvgg_B3(),
        'shufflenet_v2_x2_0_augmentation': shufflenet_v2_x2_0(),
        'shufflenetv2_2.0_augmix': shufflenet_v2_x2_0(),
        'shufflenet_v2_x2_0_ema': shufflenet_v2_x2_0(),
        'shufflenet_v2_x2_0_label_smooth': shufflenet_v2_x2_0(),
        'shufflenetv2_2.0_adamw': shufflenet_v2_x2_0(),
        "efficientnet_b0": efficientnet_b0(),
        'efficientnet_b0_nodrop': efficientnet_b0_nodrop(),
        'efficientnet_b1_nodrop_240': efficientnet_b1_nodrop(),
        'efficientnet_b2_nodrop_260': efficientnet_b2_nodrop(),
        'efficientnet_b3_nodrop_300': efficientnet_b3_nodrop(),
        'efficientnet_b4_nodrop_380': efficientnet_b4_nodrop(),
        'mobilenet_v3_large_x1_4_augmentation': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_augmix': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_ema': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_label_smooth': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_adv_train': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_adamw': mobilenet_v3(scale=1.4, dropout=0.0, mode='large'),
        'mobilenet_v3_large_x1_4_dropout': mobilenet_v3(scale=1.4, dropout=0.2, mode='large'),
        '21k_resnet50': resnet50_official(),
        '21k_vit_base_patch16_224': vit_b16_224(drop_path=0.0, dropout=0.0, attention_dropout=0.0, qkv_bias=True,
                                   representation_size=768),
        'vit_base_patch16_224_withdrop': vit_b16_224(drop_path=0.1, qkv_bias=True,
                                   representation_size=768),
        'mixer_B16_224_augmentation':mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_b16_224_augmix': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_ema': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_label_smooth.pth.tar': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_B16_224_adv_train': mixer_b16_224(drop_path=0.0, drop_path_rate=0.0),
        'mixer_b16_224_withdrop': mixer_b16_224(drop_path=0.1, drop_path_rate=0.1),

    }




def model_entry(config):
    if config['type'] not in globals():
        if config['type'].startswith('spring_'):
            try:
                from spring.models import SPRING_MODELS_REGISTRY
            except ImportError:
                print('Please install Spring2 first!')
            model_name = config['type'][len('spring_'):]
            config['type'] = model_name
            return SPRING_MODELS_REGISTRY.build(config)

    return globals()[config['type']](**config['kwargs'])


get_model = model_entry
