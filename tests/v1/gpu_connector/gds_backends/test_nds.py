# SPDX-License-Identifier: Apache-2.0
"""Tests for the NDS GDS backend (mocked unit tests; optional hardware)."""

# Standard
import os
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

from lmcache.v1.gpu_connector._gds_backends import available_backends, create_backend
from lmcache.v1.gpu_connector.gds_backends.base import Submission
from lmcache.v1.gpu_connector.gds_backends.fiemap import FileExtent, FileExtentMap
from lmcache.v1.gpu_connector.gds_backends.nds import Backend, NDSHandle


def _mock_npu_for_stream(stream_id: int):
    stream = MagicMock()
    stream.npu_stream = stream_id
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


def _immediate_read_async(*args, **kwargs):
    result = args[5] if len(args) > 5 else kwargs["result"]
    result.value = args[2] if len(args) > 2 else kwargs["size"]


def _deferred_read_async(deferred: list):
    def _fn(*args, **kwargs):
        result = args[5] if len(args) > 5 else kwargs["result"]
        size = args[2] if len(args) > 2 else kwargs["size"]
        deferred.append(lambda: setattr(result, "value", size))

    return _fn


def _immediate_write_async(mock_lib):
    def _fn(dev_handle, buf_base, segments, result, stream, on_complete=None):
        for device_off, seg_len, ptr_off in segments:
            mock_lib.write(dev_handle, buf_base, seg_len, device_off, ptr_off)
        if on_complete is not None:
            on_complete()
        result.value = sum(seg_len for _, seg_len, _ in segments)

    return _fn


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
def test_register_buffer_accepts_second_device(mock_library):
    backend = Backend()
    lib = mock_library.return_value
    lib.capabilities.return_value.requires_buffer_registration = False
    buffers = []
    for index, ptr in ((0, 0x1000), (1, 0x2000)):
        buf = MagicMock()
        buf.device.type = "npu"
        buf.device.index = index
        buf.is_contiguous.return_value = True
        buf.view.return_value = buf
        buf.__getitem__.return_value = buf
        buf.numel.return_value = 4096
        buf.data_ptr.return_value = ptr
        buffers.append(buf)
    backend.register_buffer(buffers[0])
    backend.register_buffer(buffers[1])
    assert [reg.base_ptr for reg in backend._regions] == [0x1000, 0x2000]
    assert [reg.device_index for reg in backend._regions] == [0, 1]


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_register_buffer_unregisters_pinned_regions_on_failure(mock_library):
    backend = Backend()
    region_size = 16 * 1024 * 1024
    chunks = []
    for base in (0x1000, 0x1000 + region_size):
        chunk = MagicMock()
        chunk.numel.return_value = region_size
        chunk.data_ptr.return_value = base
        chunks.append(chunk)
    view = MagicMock()
    view.numel.return_value = region_size * 2
    view.__getitem__.side_effect = chunks
    buf = MagicMock()
    buf.device.type = "npu"
    buf.device.index = 0
    buf.is_contiguous.return_value = True
    buf.view.return_value = view
    lib = mock_library.return_value
    lib.capabilities.return_value.requires_buffer_registration = True
    lib.register_memory.side_effect = [42, OSError(1, "pin failed")]

    with pytest.raises(OSError, match="pin failed"):
        backend.register_buffer(buf)

    lib.unregister_memory.assert_called_once_with(0x1000, region_size)
    assert backend._regions == []
    assert backend._region_base_ptrs == []


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_nds_handle_read_async_sets_submission(mock_library):
    mock_library.return_value.read_async.side_effect = _immediate_read_async
    mock_library.return_value.capabilities.return_value = MagicMock(can_write=False)
    backend = Backend()
    backend._slab_size_bytes = 1 << 20
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 1, MagicMock(), "/p", slab_size=backend._slab_size_bytes)
    sub = handle.read_async(0x1000, 4096, 0, 0, 7)
    assert isinstance(sub, Submission)
    assert sub.bytes_done == 4096
    mock_library.return_value.read_async.assert_called_once()
    call = mock_library.return_value.read_async.call_args
    assert call.args[0] == handle._handle
    assert call.args[5] is sub.result
    assert call.args[6] == 7


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_nds_handle_read_async_defers_until_library_callback(mock_library):
    deferred: list = []
    mock_library.return_value.read_async.side_effect = _deferred_read_async(deferred)
    backend = Backend()
    backend._slab_size_bytes = 4096
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 1, MagicMock(), "/p", slab_size=4096)
    sub = handle.read_async(0x1000, 4096, 0, 0, 7)
    assert sub.bytes_done == 0
    deferred[0]()
    assert sub.bytes_done == 4096


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_nds_handle_read_async_forwards_stream_and_result(mock_library):
    mock_library.return_value.read_async.side_effect = _immediate_read_async
    backend = Backend()
    backend._slab_size_bytes = 4096
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 11, MagicMock(), "/p", slab_size=4096)
    sub = handle.read_async(0x1000, 4096, 0, 0, 7)
    mock_library.return_value.read.assert_not_called()
    mock_library.return_value.read_async.assert_called_once_with(
        handle._handle, 0x1000, 4096, 0, 0, sub.result, 7
    )


def test_nds_handle_close_idempotent():
    backend = MagicMock()
    backend.deregister_handle = MagicMock()
    backend._library.return_value.wait_callbacks = MagicMock()
    handle = NDSHandle(backend, 5, MagicMock(), "/p", slab_size=4096)
    handle._dev_fd = 6
    handle._dev_handle = MagicMock()
    with patch("os.close") as close:
        handle.close()
        handle.close()
        assert close.call_count == 2
    assert backend._library.return_value.wait_callbacks.call_count == 2


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_nds_handle_write_async_forwards_segments_and_on_complete(mock_library):
    mock_library.return_value.write.return_value = 4096
    mock_library.return_value.capabilities.return_value = MagicMock(can_write=True)
    mock_library.return_value.write_async.side_effect = _immediate_write_async(
        mock_library.return_value
    )
    backend = Backend()
    backend._slab_size_bytes = 8192
    backend._registered_streams.add(7)
    _register_region(backend)
    handle = NDSHandle(backend, 1, MagicMock(), "/p", slab_size=8192)
    handle._dev_fd = 99
    handle._dev_handle = 88
    handle._extents = FileExtentMap(
        (
            FileExtent(logical=0, physical=0xA000, length=4096, flags=0),
            FileExtent(logical=4096, physical=0xB000, length=4096, flags=0),
        ),
        8192,
    )
    handle._ensure_extents_initialized = MagicMock()  # type: ignore[method-assign]
    with patch("os.posix_fadvise") as fadvise:
        sub = handle.write_async(0x1000, 8192, 0, 0, 7)
    assert sub.bytes_done == 8192
    wcall = mock_library.return_value.write_async.call_args
    assert wcall.args[0] == 88
    segments = wcall.args[2]
    assert segments[0][0] == 0xA000 and segments[1][0] == 0xB000
    assert segments[0][2] == 0 and segments[1][2] == 4096
    assert wcall.kwargs["on_complete"] is not None
    fadvise.assert_called_once_with(handle._fd, 0, 8192, os.POSIX_FADV_DONTNEED)
    assert mock_library.return_value.write.call_count == 2


@patch("lmcache.v1.gpu_connector.gds_backends.nds.Backend._library")
def test_nds_write_segments_use_buffer_offset_not_file_offset(mock_library):
    mock_library.return_value.write.return_value = 4096
    mock_library.return_value.capabilities.return_value = MagicMock(can_write=True)
    mock_library.return_value.write_async.side_effect = _immediate_write_async(
        mock_library.return_value
    )
    backend = Backend()
    backend._slab_size_bytes = 16384
    backend._registered_streams.add(7)
    _register_region(backend, nbytes=8192)
    handle = NDSHandle(backend, 1, MagicMock(), "/p", slab_size=16384)
    handle._dev_fd = 99
    handle._dev_handle = 88
    handle._extents = FileExtentMap(
        (FileExtent(logical=0, physical=0xA000, length=16384, flags=0),),
        16384,
    )
    handle._ensure_extents_initialized = MagicMock()  # type: ignore[method-assign]
    with patch("os.posix_fadvise"):
        handle.write_async(0x1000, 4096, 8192, 0, 7)
    segments = mock_library.return_value.write_async.call_args.args[2]
    assert segments == [(0xA000 + 8192, 4096, 0)]


def test_nds_handle_close_idempotent_file_only():
    backend = MagicMock()
    backend.deregister_handle = MagicMock()
    backend._library.return_value.wait_callbacks = MagicMock()
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
