"""BEiT-3 retrieval architecture (inference-only, text tower included).

Trimmed from microsoft/unilm/beit3 (`modeling_finetune.py`, `modeling_utils.py`),
keeping only what is needed to encode text/image into the shared 1024-d
retrieval space at inference time. Training-only pieces (the ClipLoss
criterion, timm's model registry) are intentionally left out.

The state_dict layout (`beit3.*`, `language_head.*`, `vision_head.*`,
`logit_scale`) is unchanged from the upstream `BEiT3ForRetrieval`, so a real
`beit3_large_patch16_384_retrieval` checkpoint loads into this module with
zero missing/unexpected keys.

Upstream source: https://github.com/microsoft/unilm/tree/master/beit3
(MIT License, Copyright (c) 2023 Microsoft)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchscale.architecture.config import EncoderConfig
from torchscale.model.BEiT3 import BEiT3


def build_large_retrieval_config(img_size: int = 384, vocab_size: int = 64010) -> EncoderConfig:
    """Build the `beit3_large_patch16_384_retrieval` encoder config.

    Mirrors `modeling_utils._get_large_config(img_size=384, ...)` from the
    upstream repo exactly, so the resulting module shapes match the public
    BEiT3-large retrieval checkpoint.
    """
    return EncoderConfig(
        img_size=img_size,
        patch_size=16,
        vocab_size=vocab_size,
        multiway=True,
        layernorm_embedding=False,
        normalize_output=True,
        no_output_layer=True,
        drop_path_rate=0,
        encoder_embed_dim=1024,
        encoder_attention_heads=16,
        encoder_ffn_embed_dim=1024 * 4,
        encoder_layers=24,
        checkpoint_activations=False,
    )


class BEiT3ForRetrieval(nn.Module):
    """Dual encoder (vision_head / language_head) over a shared BEiT3 backbone."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.args = config
        self.beit3 = BEiT3(config)
        embed_dim = config.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        image: torch.Tensor | None = None,
        text_description: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        only_infer: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not only_infer:
            raise NotImplementedError(
                "This trimmed BEiT3ForRetrieval only supports inference (only_infer=True)."
            )

        vision_cls = None
        if image is not None:
            outputs = self.beit3(textual_tokens=None, visual_tokens=image, text_padding_position=None)
            vision_cls = F.normalize(self.vision_head(outputs["encoder_out"][:, 0, :]), dim=-1)

        language_cls = None
        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            language_cls = F.normalize(self.language_head(outputs["encoder_out"][:, 0, :]), dim=-1)

        return vision_cls, language_cls
