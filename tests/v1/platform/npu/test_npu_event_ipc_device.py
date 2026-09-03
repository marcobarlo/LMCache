# SPDX-License-Identifier: Apache-2.0
"""NPU-gated tests: cross-process interprocess events and device staging."""

# Standard
import multiprocessing as mp

# Third Party
import pytest
import torch

# First Party
import lmcache.lmcache_native as lmcache_native
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.platform.npu.cache_context import _TempNpuBuffer

pytestmark = [
    pytest.mark.npu,
    pytest.mark.no_shared_allocator,
]

pytestmark_skip = not (hasattr(torch, "npu") and torch.npu.is_available())

requires_npu = pytest.mark.skipif(
    pytestmark_skip, reason="Ascend NPU hardware is required"
)


def _child_event_producer(conn) -> None:
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        payload = torch.full((1024,), 7.0, device="npu")
        payload.mul_(2)
    event = torch.npu.Event(enable_timing=False, interprocess=True)
    event.record(stream)
    conn.send({"handle": event.ipc_handle(), "expected": 14.0})
    conn.recv()  # keep the producer alive until the parent is done


@requires_npu
def test_cross_process_event_import_and_sync() -> None:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=_child_event_producer, args=(child_conn,))
    process.start()
    try:
        message = parent_conn.recv()
        imported = torch.npu.Event.from_ipc_handle(
            torch.device("npu:0"), message["handle"]
        )
        imported.synchronize()
        assert imported.query() is True
    finally:
        parent_conn.send("done")
        process.join(timeout=30)
    assert process.exitcode == 0


@requires_npu
def test_backend_roundtrip_on_device() -> None:
    from lmcache.v1.platform.npu import NpuDeviceSpec

    backend = NpuDeviceSpec().event_ipc_backend
    device = torch.device("npu:0")
    backend.check_event_support(device)
    event = backend.create_event(device)
    stream = torch.npu.Stream()
    payload = torch.ones(8, device=device)
    payload.add_(1)
    backend.record_event(event, stream)
    backend.synchronize_event(event, device)
    assert backend.query_event(event) is True
    handle = backend.export_event(event, device)
    # Same-process ``from_ipc_handle`` fails with driver error 17 (ACL
    # 507899) on this CANN build, so handle import is verified across
    # processes in ``test_cross_process_event_import_and_sync``.
    assert isinstance(handle, bytes) and len(handle) > 0


@requires_npu
def test_temp_buffer_allocates_on_device() -> None:
    device = torch.device("npu:0")
    num_layers, num_blocks, block_size, width = 4, 8, 16, 576
    tensors = [
        torch.zeros(num_blocks, block_size, width, device=device)
        for _ in range(num_layers)
    ]
    manager = KVLayerGroupsManager(
        tensors,
        engine_kv_formats=([lmcache_native.EngineKVFormat.NL_X_NB_BS_HS] * num_layers),
        engine_group_infos=(),
        lmcache_tokens_per_chunk=256,
    )
    buffer = _TempNpuBuffer(
        kv_layer_groups_manager=manager,
        lmcache_tokens_per_chunk=256,
        device=device,
        max_batch_size=4,
    )
    view = buffer.get_temp_kernel_group_buffer(0, 0)
    assert view.device.type == "npu"
    view.fill_(1.0)
    flat = buffer.get_temp_object_group_buffer(0, 0)
    assert flat.view(torch.float32).sum().item() == float(view.numel())


@requires_npu
def test_pointer_mode_memcpy_roundtrip_on_device() -> None:
    """lmcache_memcpy_async pointer mode copies via tensor views, not CUDA."""
    # Third Party
    import lmcache.lmcache_native as lmcache_native

    from lmcache.v1.platform.npu.device_ops import NpuDeviceOps

    ops = NpuDeviceOps()
    device = torch.device("npu:0")
    # int32: aclnnArange does not implement uint8 on this CANN build.
    src = torch.arange(64, dtype=torch.int32, device=device)
    host = torch.zeros(64, dtype=torch.int32, pin_memory=True)
    ops.lmcache_memcpy_async(
        host.data_ptr(),
        src.data_ptr(),
        256,
        lmcache_native.TransferDirection.D2H,
        0,
        1,
    )
    torch.npu.current_stream().synchronize()
    assert torch.equal(host, src.cpu())

    mutated = torch.full((64,), 7, dtype=torch.int32)
    ops.lmcache_memcpy_async(
        src.data_ptr(),
        mutated.data_ptr(),
        256,
        lmcache_native.TransferDirection.H2D,
        0,
        1,
    )
    torch.npu.current_stream().synchronize()
    assert torch.equal(src.cpu(), mutated)
