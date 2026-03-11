"""
LLM model abstraction: HuggingFace open-source model (Mistral/Llama), swappable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.logger import get_logger


class LLM(Protocol):
    """Minimal interface used by the RAG pipeline."""

    def invoke(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class LLMConfig:
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.3
    do_sample: bool = True
    device_map: str = "auto"
    torch_dtype: str = "auto"  # 'auto' | 'float16' | 'bfloat16' | 'float32'


def _resolve_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype}")
    return mapping[dtype]


def get_llm(model_name: Optional[str] = None, **kwargs: Any) -> LLM:
    """
    Return a configured HuggingFace LLM wrapped for LangChain usage.

    Designed to be modular and swappable later.

    Args:
        model_name: HuggingFace model id (e.g. mistralai/Mistral-7B-Instruct-v0.2).
        **kwargs: Supported keys include max_new_tokens, temperature, do_sample,
            device_map, torch_dtype.

    Returns:
        An object with `.invoke(prompt: str) -> str`.
    """
    logger = get_logger()
    if not model_name:
        raise ValueError("model_name is required")

    cfg = LLMConfig(model_name=model_name, **{k: v for k, v in kwargs.items() if k in LLMConfig.__annotations__})
    logger.info(f"Loading LLM: {cfg.model_name}")

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model_kwargs: Dict[str, Any] = {"device_map": cfg.device_map}
    dtype = _resolve_dtype(cfg.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=int(cfg.max_new_tokens),
        temperature=float(cfg.temperature),
        do_sample=bool(cfg.do_sample),
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen)
