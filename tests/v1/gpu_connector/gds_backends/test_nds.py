# SPDX-License-Identifier: Apache-2.0
"""Tests for the NDS GDS backend (mocked unit tests; optional hardware)."""

# Standard
import os
import threading
import time
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

from lmcache.v1.gpu_connector._gds_backends import available_backends, create_backend
from lmcache.v1.gpu_connector.gds_backends.base import Submission
from lmcache.v1.gpu_connector.gds_backends.nds import Backend, NDSHandle, _SlabExtent


def _run_host_callback_immediately(raw_stream: int, callback) -> None:
    callback()


def _append_host_callback(raw_stream: int, callback, captured: list) -> None:
    captured.append(callback)


def _mock_npu_for_stream(stream_id: int, launch_host_func):
    stream = MagicMock()
    stream.npu_stream = stream_id
    stream.launch_host_func = launch_host_func
    npu_mod = MagicMock()
    npu_mod.is_available.return_value = True
    npu_mod.current_stream.return_value = stream
    return npu_mod


def _register_region(backend: Backend, base_ptr: int = 0x1000, nbytes: int = 8192) -> None:
    backend._regions.append(
        __import__(
            "lmcache.v1.gpu_connector.gds_backends.nds", fromlist=["_RegisteredRegion"]
        )._RegisteredRegion(
            tensor=MagicMock(),
            base_ptr=base_ptr,
            nbytes=nbytes,
            device_index=0,
        )
    )
    backend._region_base_ptrs.append(base_ptr)


def test_nds_listed_in_available_backends():
    assert "nds" in available_backends()


def test_create_backend_nds_requires_npu(monkeypatch):
    monkeypatch.setattr(torch, "npu", None, raising=False)
    with pytest.raises(ValueError, match="torch.npu"):
        create_backend("nds")


def test_nds_is_default_when_npu_available(monkeypatch):
    npu = MagicMock()
    npu.is_available.return_value = True
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    assert Backend.is_default()


def test_nds_is_not_default_without_npu(monkeypatch):
    monkeypatch.setattr(torch, "npu", None, raising=False)
    assert not Backend.is_default()


@patch("lmcache.v1.gpu_connector.gds_backends.nds._sync_transfer_stream")
def test_synchronize_for_stream_uses_stream_pointer(mock_sync):
    backend = Backend()
    backend._registered_streams.add(0xABCD)
    npu_mod = MagicMock()
    npu_mod.is_available.return_value = True
    with patch.object(torch, "npu", npu_mod, create=True):
        backend.synchronize_for_stream(0xABCD)
    mock_sync.assert_called_once_with(0xABCD)


def test_synchronize_for_stream_rejects_unregistered_stream():
    backend = Backend()
    with pytest.raises(RuntimeError, match="not registered"):
        backend.synchronize_for_stream(99)


@patch("lmcache.v1.gpu_connector.gds_backends.nds._sync_transfer_stream")
def test_synchronize_for_stream_does_not_call_device_synchronize(mock_sync):
    backend = Backend()
    backend._registered_streams.add(1)
    npu_mod = MagicMock()
    npu_mod.is_available.return_value = True
    npu_mod.synchronize = MagicMock()
    with patch.object(torch, "npu", npu_mod, create=True):
        backend.synchronize_for_stream(1)
        npu_mod.synchronize.assert_not_called()
    mock_sync.assert_called_once_with(1)


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_backend_open_handle_closes_fd_on_register_failure(mock_library):
    mock_library.return_value.file_register.side_effect = RuntimeError("register")
    backend = Backend()
    backend._ensure_driver_open = MagicMock()  # type: ignore[method-assign]
    with patch("os.close") as close:
        with pytest.raises(RuntimeError, match="register"):
            backend.open_handle(42, "/tmp/x")
        close.assert_called_once_with(42)


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_register_buffer_rejects_cpu(mock_library):
    backend = Backend()
    buf = torch.empty(4096, dtype=torch.uint8)
    with pytest.raises(ValueError, match="NPU"):
        backend.register_buffer(buf)


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_register_buffer_rejects_second_device(mock_library):
    backend = Backend()
    backend._bound_device_index = 0
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu not present")
    t0 = MagicMock()
    t0.device.type = "npu"
    t0.device.index = 0
    t0.is_contiguous.return_value = True
    t0.view.return_value = t0
    t0.numel.return_value = 4096
    t0.data_ptr.return_value = 0x1000
    t1 = MagicMock()
    t1.device.type = "npu"
    t1.device.index = 1
    t1.is_contiguous.return_value = True
    t1.view.return_value = t1
    t1.numel.return_value = 4096
    t1.data_ptr.return_value = 0x2000
    backend.register_buffer(t0)
    with pytest.raises(ValueError, match="one NPU"):
        backend.register_buffer(t1)


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
@patch(
    "lmcache.v1.gpu_connector.gds_backends.nds._launch_nds_host_callback",
    side_effect=_run_host_callback_immediately,
)
def test_nds_handle_read_async_sets_submission(mock_launch, mock_library):
    mock_library.return_value.read.return_value = 4096
    mock_library.return_value.capabilities.return_value = MagicMock(can_write=False)
    backend = Backend()
    backend._slab_size_bytes = 1 << 20
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 1, MagicMock(), "/p", slab_size=backend._slab_size_bytes)
    npu_mod = _mock_npu_for_stream(7, MagicMock())
    with patch.object(torch, "npu", npu_mod, create=True):
        sub = handle.read_async(0x1000, 4096, 0, 0, 7)
    assert isinstance(sub, Submission)
    assert sub.bytes_done == 4096


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_nds_handle_read_async_defers_until_callback(mock_library):
    mock_library.return_value.read.return_value = 4096
    captured: list = []

    def capture(raw_stream: int, callback) -> None:
        _append_host_callback(raw_stream, callback, captured)

    backend = Backend()
    backend._slab_size_bytes = 4096
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 1, MagicMock(), "/p", slab_size=4096)
    npu_mod = _mock_npu_for_stream(7, MagicMock())
    with patch.object(torch, "npu", npu_mod, create=True):
        with patch(
            "lmcache.v1.gpu_connector.gds_backends.nds._launch_nds_host_callback",
            side_effect=capture,
        ):
            sub = handle.read_async(0x1000, 4096, 0, 0, 7)
    assert sub.bytes_done == 0
    mock_library.return_value.read.assert_not_called()
    captured[0]()
    assert sub.bytes_done == 4096


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_read_async_rejects_wrong_current_stream(mock_library):
    backend = Backend()
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 1, MagicMock(), "/p", slab_size=4096)
    npu_mod = _mock_npu_for_stream(8, MagicMock())
    with patch.object(torch, "npu", npu_mod, create=True):
        with pytest.raises(RuntimeError, match="current NPU stream"):
            handle.read_async(0x1000, 4096, 0, 0, 7)


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_read_async_uses_acl_host_func_when_stream_lacks_launch(mock_library):
    mock_library.return_value.read.return_value = 4096
    backend = Backend()
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 1, MagicMock(), "/p", slab_size=4096)
    stream = MagicMock()
    stream.npu_stream = 7
    del stream.launch_host_func
    npu_mod = MagicMock()
    npu_mod.is_available.return_value = True
    npu_mod.current_stream.return_value = stream
    acl_lib = MagicMock()
    acl_lib.aclrtLaunchHostFunc.return_value = 0
    with patch.object(torch, "npu", npu_mod, create=True):
        with patch(
            "lmcache.v1.gpu_connector.gds_backends.nds._acl_rt_library",
            return_value=acl_lib,
        ):
            with patch(
                "lmcache.v1.gpu_connector.gds_backends.nds._launch_nds_host_callback",
                side_effect=_run_host_callback_immediately,
            ):
                sub = handle.read_async(0x1000, 4096, 0, 0, 7)
    assert sub.bytes_done == 4096


def test_nds_handle_close_idempotent():
    backend = MagicMock()
    backend.deregister_handle = MagicMock()
    handle = NDSHandle(backend, 5, MagicMock(), "/p", slab_size=4096)
    handle._dev_fd = 6
    handle._dev_handle = MagicMock()
    with patch("os.close") as close:
        handle.close()
        handle.close()
        assert close.call_count == 2


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_close_waits_for_inflight_callback(mock_library):
    mock_library.return_value.read.return_value = 4096
    captured: list = []

    def capture(raw_stream: int, callback) -> None:
        _append_host_callback(raw_stream, callback, captured)

    backend = Backend()
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 5, MagicMock(), "/p", slab_size=4096)
    handle._dev_fd = 6
    handle._dev_handle = MagicMock()
    deregister = MagicMock(wraps=backend.deregister_handle)
    backend.deregister_handle = deregister  # type: ignore[method-assign]
    npu_mod = _mock_npu_for_stream(7, MagicMock())
    with patch.object(torch, "npu", npu_mod, create=True):
        with patch(
            "lmcache.v1.gpu_connector.gds_backends.nds._launch_nds_host_callback",
            side_effect=capture,
        ):
            handle.read_async(0x1000, 4096, 0, 0, 7)
    closed = threading.Event()

    def close_in_thread() -> None:
        handle.close()
        closed.set()

    thread = threading.Thread(target=close_in_thread)
    thread.start()
    time.sleep(0.05)
    assert not closed.is_set()
    captured[0]()
    thread.join(timeout=2.0)
    assert closed.is_set()
    assert deregister.call_count >= 1


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
@patch(
    "lmcache.v1.gpu_connector.gds_backends.nds._launch_nds_host_callback",
    side_effect=_run_host_callback_immediately,
)
def test_nds_handle_write_async_uses_block_device_offset(
    mock_launch, mock_library
):
    mock_library.return_value.write.return_value = 4096
    mock_library.return_value.capabilities.return_value = MagicMock(can_write=True)
    backend = Backend()
    backend._slab_size_bytes = 8192
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 1, MagicMock(), "/p", slab_size=8192)
    handle._dev_fd = 99
    handle._dev_handle = 88
    handle._extents = [
        _SlabExtent(logical=0, physical=0xA000, length=4096, flags=0),
        _SlabExtent(logical=4096, physical=0xB000, length=4096, flags=0),
    ]
    handle._ensure_extents_initialized = MagicMock()  # type: ignore[method-assign]
    npu_mod = _mock_npu_for_stream(7, MagicMock())
    with patch.object(torch, "npu", npu_mod, create=True):
        sub = handle.write_async(0x1000, 8192, 0, 0, 7)
    assert sub.bytes_done == 8192
    assert mock_library.return_value.write.call_count == 2
    first = mock_library.return_value.write.call_args_list[0]
    second = mock_library.return_value.write.call_args_list[1]
    assert first.args[0] == 88
    assert first.args[3] == 0xA000
    assert first.args[4] == 0
    assert second.args[3] == 0xB000
    assert second.args[4] == 4096
    mock_library.return_value.read.assert_not_called()


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
@patch(
    "lmcache.v1.gpu_connector.gds_backends.nds._launch_nds_host_callback",
    side_effect=_run_host_callback_immediately,
)
def test_nds_handle_read_async_uses_file_fd(mock_launch, mock_library):
    mock_library.return_value.read.return_value = 4096
    backend = Backend()
    backend._slab_size_bytes = 4096
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 11, MagicMock(), "/p", slab_size=4096)
    npu_mod = _mock_npu_for_stream(7, MagicMock())
    with patch.object(torch, "npu", npu_mod, create=True):
        handle.read_async(0x1000, 4096, 0, 0, 7)
    mock_library.return_value.read.assert_called_once()
    assert mock_library.return_value.read.call_args.args[0] == handle._handle
    mock_library.return_value.write.assert_not_called()


def test_nds_handle_close_idempotent_file_only():
    backend = MagicMock()
    backend.deregister_handle = MagicMock()
    handle = NDSHandle(backend, 5, MagicMock(), "/p", slab_size=4096)
    with patch("os.close") as close:
        handle.close()
        handle.close()
        assert close.call_count == 1


@pytest.mark.skipif(
    not os.environ.get("NDS_TEST_DIR") or not getattr(torch, "npu", None),
    reason="hardware test needs NDS_TEST_DIR and torch.npu",
)
def test_nds_handle_reads_slab_without_lmcache():
    pytest.importorskip("lmcache_ascend")
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    backend = create_backend("nds")
    try:
        slab_dir = os.environ["NDS_TEST_DIR"]
        h = backend.open_slab(slab_dir, 4096, direct_io=False)
        buf = torch.empty(4096, dtype=torch.uint8, device="npu:0")
        stream = torch.npu.current_stream().npu_stream
        backend.register_stream(stream)
        backend.register_buffer(buf)
        sub = h.read_async(buf.data_ptr(), 4096, 0, 0, stream)
        torch.npu.synchronize()
        assert sub.bytes_done == 4096
        h.close()
    finally:
        backend.close_driver()


@pytest.mark.skipif(
    not os.environ.get("NDS_TEST_DIR") or not getattr(torch, "npu", None),
    reason="hardware test needs NDS_TEST_DIR and torch.npu",
)
def test_nds_handle_roundtrip_without_lmcache():
    pytest.importorskip("lmcache_ascend")
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    from lmcache_ascend.v1.gds.nds_library import get_nds_library

    lib = get_nds_library()
    lib.load()
    if not lib.capabilities().can_write:
        pytest.skip("NDS write symbol not available")
    backend = create_backend("nds")
    try:
        slab_dir = os.environ["NDS_TEST_DIR"]
        h = backend.open_slab(slab_dir, 8192, direct_io=False)
        buf = torch.empty(4096, dtype=torch.uint8, device="npu:0")
        stream = torch.npu.current_stream().npu_stream
        backend.register_stream(stream)
        backend.register_buffer(buf)
        pattern = torch.full((4096,), 0xCD, dtype=torch.uint8, device="npu:0")
        buf.copy_(pattern)
        h.write_async(buf.data_ptr(), 4096, 0, 0, stream)
        torch.npu.synchronize()
        buf.zero_()
        sub = h.read_async(buf.data_ptr(), 4096, 0, 0, stream)
        torch.npu.synchronize()
        assert sub.bytes_done == 4096
        assert torch.equal(buf.cpu(), pattern.cpu())
        h.close()
    finally:
        backend.close_driver()
