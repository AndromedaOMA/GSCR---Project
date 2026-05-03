"""
Load trained HFWrapperULMFiT checkpoints for inference without re-downloading
the base BERT from the Hub via AutoModel.from_pretrained (avoids meta-device
initialization issues in some Torch/Accelerate setups).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, BertConfig

from src.preprocess.HFWrapperULMFiT import (
    ENCODER_FROM_STRUCTURE_ONLY_ATTR,
    HFWrapperULMFiT,
)


def load_detector_for_inference(
    model_dir: str | Path,
    device: torch.device,
) -> Tuple[HFWrapperULMFiT, AutoTokenizer]:
    model_dir = Path(model_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    config = BertConfig.from_pretrained(str(model_dir))
    setattr(config, ENCODER_FROM_STRUCTURE_ONLY_ATTR, True)
    model = HFWrapperULMFiT(config)

    saf_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if saf_path.is_file():
        state = load_file(str(saf_path), device="cpu")
    elif bin_path.is_file():
        try:
            state = torch.load(str(bin_path), map_location=torch.device("cpu"), weights_only=True)
        except Exception:
            state = torch.load(str(bin_path), map_location=torch.device("cpu"))
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir} "
            "(expected model.safetensors or pytorch_model.bin)."
        )

    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model, tokenizer
