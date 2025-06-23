import logging
import timm


# def get_pretrained_supervised_encoder(pretrained_checkpoint_loc, channels, timm_kwargs={}):
#     logging.info(f"Loading {pretrained_checkpoint_loc}")
#     from zoobot.pytorch.estimators import define_model

#     return define_model.ZoobotTree.load_from_checkpoint(
#         pretrained_checkpoint_loc, channels=channels, **timm_kwargs
#     ).encoder


def get_timm_encoder(name="resnet50", channels=3, pretrained=True, timm_kwargs={}):
    logging.info(f"Loading {name} from hub, over to timm")
    model = timm.create_model(
        name,
        in_chans=channels,
        num_classes=0,
        pretrained=pretrained,
        **timm_kwargs,
    )
    logging.info('Loaded')
    return model
