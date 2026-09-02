# SPDX-License-Identifier: Apache-2.0
"""NPU cache context for LMCache-driven multiprocess transfer."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.utils import (  # noqa: F401 — used by NpuCacheContext
    get_group_data_ptrs,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

logger = init_logger(__name__)


class _NpuHostCallbackStream:
    """Adapter for stream-ordered callback call sites.

    CUDA contexts expose a CuPy stream as ``cupy_stream``. torch_npu has no
    CuPy equivalent, so this adapter provides ordered callback and
    synchronization operations over a torch_npu stream and exposes its
    pointer through the existing platform stream contract.
    """

    def __init__(self, stream: object) -> None:
        self._stream = stream

    def synchronize(self) -> None:
        """Wait for all work submitted to the wrapped NPU stream."""
        synchronize = getattr(self._stream, "synchronize", None)
        if not callable(synchronize):
            raise RuntimeError("NPU stream does not support synchronization")
        synchronize()

    @property
    def ptr(self) -> int:
        """Return the wrapped NPU stream pointer for platform call sites."""
        pointer = getattr(self._stream, "npu_stream", None)
        if pointer is None:
            raise RuntimeError("NPU stream does not expose a stream pointer")
        return int(pointer)

    def launch_host_func(self, callback: Any, arg: Any = None) -> None:
        """Schedule or run ``callback(arg)``.

        If the backend stream exposes ``launch_host_func`` directly, delegate
        to it. Otherwise synchronize the stream before running the callback on
        the current thread.
        """
        launch_host_func = getattr(self._stream, "launch_host_func", None)
        if callable(launch_host_func):
            launch_host_func(callback, arg)
            return
        self.synchronize()
        callback(arg)


class _TempNpuBuffer:
    """Owns NPU staging buffers for MP block-transfer batches."""

    def __init__(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int = 4,
    ) -> None:
        self._kv_groups_manager = kv_layer_groups_manager
        self._lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self._max_batch_size = max_batch_size

        self._temp_buffer = torch.empty(
            self._get_size_for_single_batch() * max_batch_size,
            dtype=torch.uint8,
            device=device,
        )
        self._offset_map_kernel_group_only: dict[tuple[int, int], tuple[int, int]] = {}
        self._offset_map_object_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        offset = 0
        for batch_idx in range(max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_start_offset = offset
                object_group_size = 0
                object_group = self._kv_groups_manager.object_groups[object_group_idx]
                for kernel_group_idx in object_group.kernel_group_indices:
                    size = self._get_size_for_kernel_group(kernel_group_idx)
                    self._offset_map_kernel_group_only[
                        (batch_idx, kernel_group_idx)
                    ] = (
                        offset,
                        size,
                    )
                    offset += size
                    object_group_size += size

                self._offset_map_object_group_only[(batch_idx, object_group_idx)] = (
                    object_group_start_offset,
                    object_group_size,
                )

        self._shape_cache_kernel_group: dict[int, tuple[torch.Size, torch.dtype]] = {}
        for kernel_group_idx in range(self._kv_groups_manager.num_kernel_groups):
            shape = self._get_shape_for_kernel_group(
                self._lmcache_tokens_per_chunk,
                kernel_group_idx,
            )
            group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
            self._shape_cache_kernel_group[kernel_group_idx] = (shape, group.dtype)

    @property
    def max_batch_size(self) -> int:
        """Return the number of chunks that fit in the staging buffer."""
        return self._max_batch_size

    def get_temp_kernel_group_buffer(
        self,
        batch_idx: int,
        kernel_group_idx: int,
    ) -> torch.Tensor:
        """Return a typed staging view for a batch/kernel-group pair."""
        key = (batch_idx, kernel_group_idx)
        if key not in self._offset_map_kernel_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or kernel_group_idx {kernel_group_idx}"
            )

        offset, size = self._offset_map_kernel_group_only[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._temp_buffer[offset : offset + size].view(dtype).view(shape)

    def get_temp_object_group_buffer(
        self,
        batch_idx: int,
        object_group_idx: int,
    ) -> torch.Tensor:
        """Return a flat ``uint8`` staging view for a batch/object-group pair."""
        key = (batch_idx, object_group_idx)
        if key not in self._offset_map_object_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or object_group_idx {object_group_idx}"
            )

        offset, size = self._offset_map_object_group_only[key]
        return self._temp_buffer[offset : offset + size]

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Return ``(shape, dtype)`` for a kernel group and token count."""
        _, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._get_shape_for_kernel_group(num_tokens, kernel_group_idx), dtype

    def get_cache_size_per_token(self) -> int:
        """Return total cache bytes per logical token across all groups."""
        return self._get_size_for_single_batch() // self._lmcache_tokens_per_chunk

    def _get_shape_for_kernel_group(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> torch.Size:
        if num_tokens % self._lmcache_tokens_per_chunk != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be a multiple of "
                f"lmcache_tokens_per_chunk ({self._lmcache_tokens_per_chunk})"
            )

        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        num_chunks = num_tokens // self._lmcache_tokens_per_chunk
        num_slots = (
            self._kv_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)
            * num_chunks
        )
        if group.engine_kv_format in (
            EngineKVFormat.NL_X_NB_BS_HS,
            EngineKVFormat.NL_X_TWO_X_NB_BS_HS,
        ):
            return torch.Size((group.num_layers, num_slots, group.hidden_dim_size))
        sd = group.shape_desc
        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def _get_size_for_kernel_group(self, kernel_group_idx: int) -> int:
        shape = self._get_shape_for_kernel_group(
            self._lmcache_tokens_per_chunk,
            kernel_group_idx,
        )
        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        return shape.numel() * group.dtype.itemsize

    def _get_size_for_object_group(self, object_group_idx: int) -> int:
        object_group = self._kv_groups_manager.object_groups[object_group_idx]
        return sum(
            self._get_size_for_kernel_group(kernel_group_idx)
            for kernel_group_idx in object_group.kernel_group_indices
        )

    def _get_size_for_single_batch(self) -> int:
        return sum(
            self._get_size_for_object_group(object_group_idx)
            for object_group_idx in range(self._kv_groups_manager.num_object_groups)
        )
