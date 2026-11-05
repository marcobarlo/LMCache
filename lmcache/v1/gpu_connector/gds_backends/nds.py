# SPDX-License-Identifier: Apache-2.0
"""Huawei NDS backend for LMCache GDS L1 (Ascend NPU direct storage)."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import bisect
import ctypes
import os
import stat

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends._file import FDGDSHandle, FileGDSBackend
from lmcache.v1.gpu_connector.gds_backends.base import GDSHandle, Submission
from lmcache.v1.gpu_connector.gds_backends.fiemap import FileExtentMap

_GDS_ALIGNMENT = 4096
_MAX_GDS_REGION = 16 * 1024 * 1024
_NDS_SECTOR = 512


@dataclass
class _RegisteredRegion:
    """One <=16 MiB NPU staging slice for bounds checks and optional mem register."""

    tensor: torch.Tensor
    base_ptr: int
    nbytes: int
    device_index: int


def _block_device_path_for_file(path: str) -> str:
    """Return ``/dev/...`` for the block device that hosts a file (``st_dev`` match)."""
    dev = os.stat(path).st_dev
    for node in Path("/dev").iterdir():
        try:
            st = node.stat()
        except OSError:
            continue
        if stat.S_ISBLK(st.st_mode) and st.st_rdev == dev:
            return str(node)
    raise ValueError(
        f"NDS slab file is not on a local block device: {path} "
        f"(dev {os.major(dev)}:{os.minor(dev)})"
    )


class Backend(FileGDSBackend):
    """NDS-backed file slab (RFC ``NDSBackend``); discovery name is ``Backend``."""

    name = "nds"
    _driver = SharedDriver()

    def __init__(self) -> None:
        super().__init__()
        self._bound_device_index: int | None = None
        self._registered_streams: set[int] = set()
        self._regions: list[_RegisteredRegion] = []
        self._region_base_ptrs: list[int] = []
        self._slab_size_bytes: int = 0

    @classmethod
    def is_default(cls) -> bool:
        npu = getattr(torch, "npu", None)
        return npu is not None and npu.is_available()

    def validate_environment(self) -> None:
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise ValueError("nds backend requires torch.npu")
        if not npu.is_available():
            raise ValueError("nds backend requires an available NPU")

    def open_slab(self, location: str, size: int, direct_io: bool) -> GDSHandle:
        """Create the slab file for load, then attach block-device store metadata.

        Records ``size`` on the backend, delegates file creation to
        :class:`FileGDSBackend`, and calls :meth:`NDSHandle.attach_store_device`
        so reads use the file fd and writes use FIEMAP physical offsets on the
        backing block device.
        """
        self._slab_size_bytes = size
        handle = super().open_slab(location, size, direct_io)
        if not isinstance(handle, NDSHandle):
            raise TypeError("NDS open_slab must return NDSHandle")
        handle.attach_store_device(self, size)
        return handle

    def open_handle(self, fd: int, path: str) -> "NDSHandle":
        try:
            handle = self.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        return NDSHandle(
            self,
            fd,
            handle,
            path,
            slab_size=self._slab_size_bytes,
        )

    def register_handle(self, fd: int) -> Any:
        self._ensure_driver_open()
        return self._library().file_register(fd)

    def deregister_handle(self, handle: Any) -> None:
        self._library().file_deregister(handle)

    def register_buffer(self, buf: torch.Tensor) -> None:
        """Register NPU staging memory in <=16 MiB regions; libnds when required."""
        if buf.device.type != "npu":
            raise ValueError("register_buffer: tensor must be on NPU")
        if not buf.is_contiguous():
            raise ValueError("register_buffer: tensor must be contiguous")
        device_index = int(buf.device.index or 0)
        self._bind_or_validate_device(device_index)
        view = buf.view(torch.uint8)
        inserted: list[_RegisteredRegion] = []
        try:
            for start in range(0, view.numel(), _MAX_GDS_REGION):
                region = view[start : min(start + _MAX_GDS_REGION, view.numel())]
                base = region.data_ptr()
                nbytes = region.numel()
                reg = _RegisteredRegion(
                    tensor=region,
                    base_ptr=base,
                    nbytes=nbytes,
                    device_index=device_index,
                )
                self._regions.append(reg)
                idx = bisect.bisect_left(self._region_base_ptrs, base)
                self._region_base_ptrs.insert(idx, base)
                inserted.append(reg)
                caps = self._library().capabilities()
                if caps.requires_buffer_registration:
                    self._library().register_memory(base, nbytes)
        except BaseException:
            for reg in reversed(inserted):
                idx = bisect.bisect_left(self._region_base_ptrs, reg.base_ptr)
                if (
                    idx < len(self._region_base_ptrs)
                    and self._region_base_ptrs[idx] == reg.base_ptr
                ):
                    self._region_base_ptrs.pop(idx)
                    self._regions.pop(idx)
            raise

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        """Reverse :meth:`register_buffer` for each 16 MiB region base in ``buf``."""
        view = buf.view(torch.uint8)
        for start in range(0, view.numel(), _MAX_GDS_REGION):
            region = view[start : min(start + _MAX_GDS_REGION, view.numel())]
            base = region.data_ptr()
            idx = bisect.bisect_left(self._region_base_ptrs, base)
            if (
                idx >= len(self._region_base_ptrs)
                or self._region_base_ptrs[idx] != base
            ):
                continue
            region = self._regions[idx]
            caps = self._library().capabilities()
            if caps.requires_buffer_registration:
                self._library().unregister_memory(region.base_ptr, region.nbytes)
            self._region_base_ptrs.pop(idx)
            self._regions.pop(idx)

    def register_stream(self, raw_stream: int) -> None:
        self._registered_streams.add(raw_stream)

    def deregister_stream(self, raw_stream: int) -> None:
        self._registered_streams.discard(raw_stream)

    def _bind_or_validate_device(self, device_index: int) -> None:
        """Pin this backend to a single NPU index for the process lifetime."""
        if self._bound_device_index is None:
            self._bound_device_index = device_index
            return
        if device_index != self._bound_device_index:
            raise ValueError(
                f"nds backend supports one NPU per process; "
                f"bound device {self._bound_device_index}, got {device_index}"
            )

    def _validate_stream(self, raw_stream: int) -> None:
        if raw_stream not in self._registered_streams:
            raise RuntimeError(f"NDS stream {raw_stream} is not registered")

    def _validate_io_bounds(
        self,
        file_offset: int,
        size: int,
        buf_base: int,
        buf_offset: int,
        alignment: int = _GDS_ALIGNMENT,
    ) -> None:
        """Check GDS alignment, slab span, and registered NPU region before I/O."""
        if size <= 0:
            raise ValueError("I/O size must be positive")
        if file_offset < 0 or buf_offset < 0:
            raise ValueError("I/O offsets must be non-negative")
        if file_offset % alignment or size % alignment or buf_offset % alignment:
            raise ValueError(
                f"NDS I/O requires {alignment}-byte alignment for offset and size"
            )
        if self._slab_size_bytes and file_offset + size > self._slab_size_bytes:
            raise ValueError("file I/O exceeds slab size")
        idx = bisect.bisect_left(self._region_base_ptrs, buf_base)
        if idx >= len(self._regions) or self._regions[idx].base_ptr != buf_base:
            raise RuntimeError(f"NDS buffer base 0x{buf_base:x} is not registered")
        region = self._regions[idx]
        if buf_offset + size > region.nbytes:
            raise RuntimeError("buffer I/O exceeds registered region")

    def _library(self):
        from lmcache_ascend.v1.gds.nds_library import get_nds_library

        return get_nds_library()

    def _open_driver(self) -> None:
        self._library().driver_open()

    def _close_driver(self) -> None:
        self._library().driver_close()


class NDSHandle(FDGDSHandle):
    """Owning NDS file handle with stream-ordered direct read/write."""

    def __init__(
        self,
        backend: Backend,
        fd: int,
        handle: Any,
        path: str,
        *,
        slab_size: int,
    ) -> None:
        super().__init__(backend, fd, handle, path)
        self._backend: Backend = backend
        self._slab_size = slab_size
        self._extents: FileExtentMap | None = None
        self._dev_fd = -1
        self._dev_handle: Any = None
        self._dev_path = ""

    def attach_store_device(self, backend: Backend, slab_size: int) -> None:
        """Prepare NDS store: FIEMAP map, block ``/dev`` open, and ``nds_register_fs``.

        Load path keeps using the slab **file** fd registered in ``open_handle``.
        Store path issues ``PWRITE`` on a separate **block device** fd at each
        extent's physical byte offset (logical ``file_offset`` is translated via
        :meth:`_write_segments`). On ``register_handle`` failure the block fd is
        closed; the file handle is unchanged (see :meth:`Backend.open_slab`).
        """
        self._slab_size = slab_size
        self._extents = FileExtentMap.load(self._fd, slab_size)
        self._dev_path = _block_device_path_for_file(self.path)
        dev_fd = os.open(self._dev_path, os.O_RDWR)
        try:
            dev_handle = backend.register_handle(dev_fd)
        except Exception:
            os.close(dev_fd)
            raise
        self._dev_fd = dev_fd
        self._dev_handle = dev_handle

    def close(self) -> None:
        self._backend._library().wait_callbacks()
        if self._dev_fd >= 0:
            dev_fd = self._dev_fd
            dev_handle = self._dev_handle
            self._dev_fd = -1
            self._dev_handle = None
            try:
                self._backend.deregister_handle(dev_handle)
            finally:
                os.close(dev_fd)
        super().close()

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """GDS ``read_async`` contract; libnds I/O runs in a stream host callback.

        Returns a :class:`Submission` immediately. ``bytes_done`` is zero until
        the callback finishes ``nds_io_submit`` / ``getevents``. Later work
        queued on the same stream runs after the callback.
        """
        self._backend._validate_stream(raw_stream)
        self._backend._validate_io_bounds(file_offset, size, buf_base, buf_offset)
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        lib = self._backend._library()
        lib.read_async(
            self._handle,
            buf_base,
            size,
            file_offset,
            buf_offset,
            sub.result,
            raw_stream,
        )
        return sub

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """GDS ``write_async`` contract; libnds store runs in a stream host callback.

        Logical ``file_offset`` is mapped to physical device offsets on the caller.
        ``bytes_done`` is zero until the callback completes all extent ``PWRITE``s.
        """
        caps = self._backend._library().capabilities()
        if not caps.can_write:
            raise RuntimeError("NDS write is not available on this platform")
        if self._dev_fd < 0:
            raise RuntimeError("NDS store device is not open on this handle")
        self._backend._validate_stream(raw_stream)
        self._backend._validate_io_bounds(file_offset, size, buf_base, buf_offset)
        self._ensure_extents_initialized(file_offset, size)
        segments = self._write_segments(file_offset, size)
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        lib = self._backend._library()

        def after_store() -> None:
            os.posix_fadvise(self._fd, file_offset, size, os.POSIX_FADV_DONTNEED)

        lib.write_async(
            self._dev_handle,
            buf_base,
            segments,
            sub.result,
            raw_stream,
            on_complete=after_store,
        )
        return sub

    def _ensure_extents_initialized(self, file_offset: int, size: int) -> None:
        """Allocate unwritten extents without leaving dirty zeros in the page cache.

        A buffered ``pwrite`` of zeros marks those pages dirty. The later NDS
        store writes the real bytes to the block device and bypasses that cache,
        so writeback of the zeros would overwrite the stored KV. ``O_DIRECT``
        zero-fill allocates the blocks and leaves the page cache clean.
        """
        end = file_offset + size
        pos = file_offset
        spans: list[tuple[int, int]] = []
        while pos < end:
            ext = self._loaded_extents().extent_at(pos)
            seg_end = min(end, ext.logical + ext.length)
            if ext.unwritten:
                spans.append((pos, seg_end - pos))
            pos = seg_end
        if not spans:
            return
        # Anonymous mmap is page-aligned. os.pwrite may copy and break
        # O_DIRECT alignment, so the write goes through libc.pwrite.
        import mmap

        chunk = 1024 * 1024
        libc = ctypes.CDLL(None, use_errno=True)
        libc.pwrite.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int64,
        ]
        libc.pwrite.restype = ctypes.c_ssize_t
        direct_fd = os.open(self.path, os.O_RDWR | os.O_DIRECT)
        try:
            buf = mmap.mmap(-1, chunk, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            view = ctypes.c_char.from_buffer(buf)
            try:
                addr = ctypes.c_void_p(ctypes.addressof(view))
                for offset, length in spans:
                    if offset % _NDS_SECTOR or length % _NDS_SECTOR:
                        raise ValueError(
                            f"NDS zero-fill range is not 512-byte aligned: "
                            f"offset={offset} length={length}"
                        )
                    done = 0
                    while done < length:
                        n = min(chunk, length - done)
                        wrote = libc.pwrite(direct_fd, addr, n, offset + done)
                        if wrote < 0:
                            err = ctypes.get_errno()
                            raise OSError(err, os.strerror(err), self.path)
                        if wrote != n:
                            raise OSError(
                                f"short O_DIRECT zero-fill: {wrote} of {n} at {offset + done}"
                            )
                        done += wrote
            finally:
                del view
                buf.close()
        finally:
            os.close(direct_fd)
        self._extents = FileExtentMap.load(self._fd, self._slab_size)

    def _loaded_extents(self) -> FileExtentMap:
        extents = self._extents
        if extents is None:
            raise RuntimeError("NDS slab extents are not loaded")
        return extents

    def _write_segments(
        self, file_offset: int, size: int
    ) -> list[tuple[int, int, int]]:
        """Return (device_offset, segment_length, logical_start) per extent piece."""
        segments: list[tuple[int, int, int]] = []
        end = file_offset + size
        pos = file_offset
        while pos < end:
            ext = self._loaded_extents().extent_at(pos)
            seg_end = min(end, ext.logical + ext.length)
            seg_len = seg_end - pos
            device_off = ext.physical + (pos - ext.logical)
            segments.append((device_off, seg_len, pos))
            pos = seg_end
        return segments
