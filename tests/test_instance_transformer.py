from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from cpa.modeling.instance_transformer import SimpleInstanceSegmentationTransformerModule


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(
            architecture="simple_instance_transformer",
            embed_dim=32,
            num_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            num_queries=4,
            patch_size=16,
            dropout=0.0,
            log_samples=1,
            visualization_threshold=0.5,
        ),
        training=SimpleNamespace(lr0=1e-4, weight_decay=1e-4, epochs=1, lrf=0.01),
    )


def test_render_sample_panel_accepts_bfloat16_predictions() -> None:
    module = SimpleInstanceSegmentationTransformerModule(_cfg())
    batch = {
        "images": torch.zeros((1, 3, 64, 64), dtype=torch.float32),
        "masks": [torch.zeros((1, 64, 64), dtype=torch.float32)],
    }
    outputs = {
        "object_logits": torch.ones((1, 4), dtype=torch.bfloat16),
        "mask_logits": torch.ones((1, 4, 8, 8), dtype=torch.bfloat16),
    }

    panel = module._render_sample_panel(batch, outputs, sample_idx=0)

    assert panel.dtype == np.uint8
    assert panel.shape == (64, 192, 3)
