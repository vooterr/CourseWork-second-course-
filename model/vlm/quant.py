from model.vlm.abstract_model import ModelConfig

import torch
from transformers import BitsAndBytesConfig


def build_load_kwargs(config):
    kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    # dtype
    if config.dtype == "float16":
        kwargs["dtype"] = torch.float16
    else:
        raise ValueError("For CUDA use float16 only")

    # quantization
    if config.quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    return kwargs