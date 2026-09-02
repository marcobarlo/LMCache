# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for NPU cache-context helpers (no Ascend hardware)."""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
import lmcache.lmcache_native as lmcache_native
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.platform.npu.cache_context import (
    _NpuHostCallbackStream,
    _TempNpuBuffer,
)

pytestmark = pytest.mark.no_shared_allocator


class _FakeStream:
    def __init__(self) -> None:
        self.npu_stream = 0x1234
        self.synchronized = 0
        self.launched: list[tuple[Any, Any]] = []

    def synchronize(self) -> None:
        self.synchronized += 1


def test_host_callback_stream_ptr_reads_npu_stream() -> None:
    adapter = _NpuHostCallbackStream(_FakeStream())
    assert adapter.ptr == 0x1234


def test_host_callback_stream_launch_runs_callback_inline_after_sync() -> None:
    stream = _FakeStream()
    adapter = _NpuHostCallbackStream(stream)
    seen: list[Any] = []
    adapter.launch_host_func(seen.append, 42)
    assert seen == [42]
    assert stream.synchronized == 1


def _mla_groups_manager() -> KVLayerGroupsManager:
    # Per-layer [NB, BS, W] planes, MLA-style single plane per layer.
    num_layers, num_blocks, block_size, width = 4, 8, 16, 576
    tensors = [torch.zeros(num_blocks, block_size, width) for _ in range(num_layers)]
    fmt = lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
    return KVLayerGroupsManager(
        tensors,
        engine_kv_formats=[fmt] * num_layers,
        engine_group_infos=(),
        lmcache_tokens_per_chunk=256,
    )


def test_temp_buffer_mla_staging_is_rank3() -> None:
    manager = _mla_groups_manager()
    buffer = _TempNpuBuffer(
        kv_layer_groups_manager=manager,
        lmcache_tokens_per_chunk=256,
        device=torch.device("cpu"),
        max_batch_size=4,
    )
    shape, dtype = buffer.get_kernel_group_shape_dtype(256, 0)
    # MLA staging is [L, slots, W] where slots are token slots: a 256-token
    # chunk with tokens_per_block=slots_per_block=16 has 256 slots.
    assert tuple(shape) == (4, 256, 576)
    assert dtype == torch.float32
    view = buffer.get_temp_kernel_group_buffer(0, 0)
    assert tuple(view.shape) == (4, 256, 576)
    flat = buffer.get_temp_object_group_buffer(0, 0)
    assert flat.dtype == torch.uint8
    assert flat.nbytes == 4 * 256 * 576 * 4


def test_temp_buffer_object_group_view_is_contiguous_union() -> None:
    manager = _mla_groups_manager()
    buffer = _TempNpuBuffer(
        kv_layer_groups_manager=manager,
        lmcache_tokens_per_chunk=256,
        device=torch.device("cpu"),
        max_batch_size=2,
    )
    kg0 = buffer.get_temp_kernel_group_buffer(1, 0)
    flat = buffer.get_temp_object_group_buffer(1, 0)
    assert kg0.data_ptr() == flat.data_ptr()
    assert buffer.max_batch_size == 2
