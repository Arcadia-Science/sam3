# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging
from contextlib import nullcontext
from typing import Union

import torch

logger = logging.getLogger(__name__)


def get_device(preferred: str = "auto") -> torch.device:
    """Return the best available torch device.

    Args:
        preferred: Explicit device string (e.g. "cuda", "mps", "cpu") or
                   "auto" to pick the best available in order
                   CUDA -> MPS -> CPU.
    """
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    """Return the preferred autocast dtype for *device*.

    CUDA  -> bfloat16  (matches existing behaviour)
    MPS   -> float16   (bfloat16 support is incomplete on MPS)
    CPU   -> bfloat16
    """
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def autocast_context(device: Union[torch.device, str], dtype=None, enabled=True):
    """Return a ``torch.autocast`` context manager appropriate for *device*.

    On MPS, autocast with ``device_type="mps"`` is still experimental and
    may not cover all ops.  When *enabled* is ``True`` but the device is MPS,
    we use ``device_type="cpu"`` as a safe no-op fallback so that inference
    still runs correctly in float32.

    Returns ``contextlib.nullcontext()`` when *enabled* is ``False``.
    """
    if not enabled:
        return nullcontext()

    if isinstance(device, str):
        device = torch.device(device)

    device_type = device.type
    if dtype is None:
        dtype = get_autocast_dtype(device)

    if device_type == "mps":
        return nullcontext()

    return torch.autocast(device_type=device_type, dtype=dtype)


def is_gpu_device(device: Union[torch.device, str]) -> bool:
    """Return True if *device* is an accelerator (CUDA or MPS)."""
    if isinstance(device, str):
        device = torch.device(device)
    return device.type in ("cuda", "mps")
