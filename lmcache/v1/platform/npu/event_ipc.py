# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU device-event IPC backend."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend


def _torch_npu_module() -> Any:
    """Return ``torch.npu``, failing closed when torch_npu is absent."""
    # Third Party
    import torch

    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError(
            "torch_npu is not installed; NPU interprocess events are unavailable."
        )
    return npu


class NpuEventIPCBackend(DefaultEventIPCBackend):
    """Event IPC backend over torch_npu interprocess events.

    torch_npu implements the CUDA-style event ABI (``interprocess=True``,
    ``ipc_handle``, ``from_ipc_handle``), so the default adapter applies
    directly; ``check_event_support`` fails closed on builds lacking it.
    """

    def __init__(self, event_module: Any | None = None) -> None:
        """Create an NPU event IPC backend.

        Args:
            event_module: Optional torch.npu-like module. Defaults to the
                installed torch_npu; injectable for tests.
        """
        super().__init__(
            event_module=(
                event_module if event_module is not None else _torch_npu_module()
            ),
            device_type="npu",
        )
