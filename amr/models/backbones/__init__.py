# Copyright (c) Meta Platforms, Inc. and affiliates.

from yacs.config import CfgNode


def create_backbone(cfg: CfgNode):
    if cfg.MODEL.BACKBONE.TYPE in ["vit"]:
        from .vit import vit

        backbone = vit(cfg)
    elif cfg.MODEL.BACKBONE.TYPE in ["vit_512_384"]:
        from .vit import vit512_384

        backbone = vit512_384(cfg)
    elif cfg.MODEL.BACKBONE.TYPE in ["vit_l"]:
        from .vit import vit_l

        backbone = vit_l(cfg)
    elif cfg.MODEL.BACKBONE.TYPE in ["vit_b"]:
        from .vit import vit_b

        backbone = vit_b(cfg)
    else:
        raise NotImplementedError("Backbone type is not implemented")

    return backbone
