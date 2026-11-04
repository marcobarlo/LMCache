# SPDX-License-Identifier: Apache-2.0
"""Huawei NDS backend for LMCache GDS L1 (Ascend NPU direct storage)."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any
import bisect
import fcntl
import os
import stat
import struct
from pathlib import Path

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends._file import FDGDSHandle, FileGDSBackend
from lmcache.v1.gpu_connector.gds_backends.base import GDSHandle, Submission

_GDS_ALIGNMENT = 4096
_MAX_GDS_REGION = 16 * 1024 * 1024
_NDS_SECTOR = 512

FS_IOC_FIEMAP = 0xC020660B
FIEMAP_EXTENT_LAST = 0x00000001
FIEMAP_EXTENT_UNKNOWN = 0x00000002
FIEMAP_EXTENT_DELALLOC = 0x00000004
FIEMAP_EXTENT_ENCODED = 0x00000008
FIEMAP_EXTENT_DATA_ENCRYPTED = 0x00000080
FIEMAP_EXTENT_NOT_ALIGNED = 0x00000100
FIEMAP_EXTENT_DATA_INLINE = 0x00000200
FIEMAP_EXTENT_DATA_TAIL = 0x00000400
FIEMAP_EXTENT_UNWRITTEN = 0x00000800
_UNSUPPORTED_FIEMAP_FLAGS = (
    FIEMAP_EXTENT_UNKNOWN
    | FIEMAP_EXTENT_DELALLOC
    | FIEMAP_EXTENT_ENCODED
    | FIEMAP_EXTENT_DATA_ENCRYPTED
    | FIEMAP_EXTENT_NOT_ALIGNED
    | FIEMAP_EXTENT_DATA_INLINE
    | FIEMAP_EXTENT_DATA_TAIL
)


@dataclass(frozen=True)
class _SlabExtent:
    logical: int
    physical: int
    length: int
    flags: int


@dataclass
class _RegisteredRegion:
    tensor: torch.Tensor
    base_ptr: int
    nbytes: int
    device_index: int


def _fiemap_extents(fd: int, size: int) -> list[_SlabExtent]:
    fm_start = 0
    out: list[_SlabExtent] = []
    while fm_start < size:
        extent_count = 32
        buf = bytearray(32 + extent_count * 40)
        query_len = size - fm_start
        struct.pack_into(
            "<QQIIII", buf, 0, fm_start, query_len, 0, 0, extent_count, 0
        )
        fcntl.ioctl(fd, FS_IOC_FIEMAP, buf, True)
        _, _, _, mapped, _, _ = struct.unpack_from("<QQIIII", buf, 0)
        if mapped == 0:
            break
        last_logical = fm_start
        for i in range(mapped):
            logical, physical, length, flags, _, _, _ = struct.unpack_from(
                "<QQQIIII", buf, 32 + i * 40
            )
            if logical + length <= 0 or length == 0:
                continue
            if logical >= size:
                continue
            trim_end = min(logical + length, size)
            trim_start = max(logical, 0)
            if trim_end <= trim_start:
                continue
            out.append(
                _SlabExtent(
                    logical=trim_start,
                    physical=physical + (trim_start - logical),
                    length=trim_end - trim_start,
                    flags=flags,
                )
            )
            last_logical = max(last_logical, logical + length)
            if flags & FIEMAP_EXTENT_LAST:
                return out
        if last_logical <= fm_start:
            break
        fm_start = last_logical
    return out


def _validate_slab_extents(extents: list[_SlabExtent], size: int) -> None:
    if not extents:
        raise ValueError("NDS slab has no FIEMAP extents")
    pos = 0
    for ext in extents:
        if ext.logical != pos:
            raise ValueError(
                f"NDS slab extents have a gap at byte {pos} (got logical {ext.logical})"
            )
        if ext.length <= 0 or ext.length % _NDS_SECTOR:
            raise ValueError(f"NDS extent length must be a positive multiple of 512: {ext}")
        if ext.physical <= 0 or ext.physical % _NDS_SECTOR:
            raise ValueError(
                f"NDS extent physical offset must be a positive multiple of 512: {ext}"
            )
        bad = ext.flags & _UNSUPPORTED_FIEMAP_FLAGS
        if bad:
            raise ValueError(f"NDS extent has unsupported FIEMAP flags 0x{bad:x}: {ext}")
        pos += ext.length
    if pos < size:
        raise ValueError(f"NDS slab extents cover {pos} bytes, need {size}")


def _block_device_path_for_file(path: str) -> str:
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
        return False

    def validate_environment(self) -> None:
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise ValueError("nds backend requires torch.npu")
        if not npu.is_available():
            raise ValueError("nds backend requires an available NPU")

    def _library(self):
        from lmcache_ascend.v1.gds.nds_library import get_nds_library

        return get_nds_library()

    def _open_driver(self) -> None:
        self._library().driver_open()

    def _close_driver(self) -> None:
        self._library().driver_close()

    def open_slab(self, location: str, size: int, direct_io: bool) -> GDSHandle:
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
                if idx < len(self._region_base_ptrs) and self._region_base_ptrs[idx] == reg.base_ptr:
                    self._region_base_ptrs.pop(idx)
                    self._regions.pop(idx)
            raise

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        view = buf.view(torch.uint8)
        for start in range(0, view.numel(), _MAX_GDS_REGION):
            region = view[start : min(start + _MAX_GDS_REGION, view.numel())]
            base = region.data_ptr()
            idx = bisect.bisect_left(self._region_base_ptrs, base)
            if idx >= len(self._region_base_ptrs) or self._region_base_ptrs[idx] != base:
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

    def synchronize_for_stream(self, raw_stream: int) -> None:
        """Wait for NPU work on the registered stream before synchronous NDS I/O."""
        self._validate_stream(raw_stream)
        if not torch.npu.is_available():
            return
        torch.npu.synchronize()


class NDSHandle(FDGDSHandle):
    """Owning NDS file handle with synchronous direct read/write."""

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
        self._extents: list[_SlabExtent] = []
        self._dev_fd = -1
        self._dev_handle: Any = None
        self._dev_path = ""

    def attach_store_device(self, backend: Backend, slab_size: int) -> None:
        """Map slab extents and register the backing block device for store."""
        self._slab_size = slab_size
        self._extents = _fiemap_extents(self._fd, slab_size)
        _validate_slab_extents(self._extents, slab_size)
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
        return self._run_sync_io(
            "read",
            buf_base,
            size,
            file_offset,
            buf_offset,
            raw_stream,
        )

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        caps = self._backend._library().capabilities()
        if not caps.can_write:
            raise RuntimeError("NDS write is not available on this platform")
        if self._dev_fd < 0:
            raise RuntimeError("NDS store device is not open on this handle")
        return self._run_sync_write(
            buf_base,
            size,
            file_offset,
            buf_offset,
            raw_stream,
        )

    def _ensure_extents_initialized(self, file_offset: int, size: int) -> None:
        end = file_offset + size
        pos = file_offset
        touched = False
        while pos < end:
            ext = self._extent_at(pos)
            seg_end = min(end, ext.logical + ext.length)
            if ext.flags & FIEMAP_EXTENT_UNWRITTEN:
                length = seg_end - pos
                os.pwrite(self._fd, b"\x00" * length, pos)
                touched = True
            pos = seg_end
        if touched:
            os.fsync(self._fd)
            self._extents = _fiemap_extents(self._fd, self._slab_size)
            _validate_slab_extents(self._extents, self._slab_size)

    def _extent_at(self, logical_offset: int) -> _SlabExtent:
        for ext in self._extents:
            if ext.logical <= logical_offset < ext.logical + ext.length:
                return ext
        raise RuntimeError(
            f"NDS slab logical offset {logical_offset} is not covered by FIEMAP"
        )

    def _write_segments(
        self, file_offset: int, size: int
    ) -> list[tuple[int, int, int]]:
        """Return (device_offset, segment_length, logical_start) per extent piece."""
        segments: list[tuple[int, int, int]] = []
        end = file_offset + size
        pos = file_offset
        while pos < end:
            ext = self._extent_at(pos)
            seg_end = min(end, ext.logical + ext.length)
            seg_len = seg_end - pos
            device_off = ext.physical + (pos - ext.logical)
            segments.append((device_off, seg_len, pos))
            pos = seg_end
        return segments

    def _run_sync_write(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        self._backend._validate_stream(raw_stream)
        self._backend._validate_io_bounds(file_offset, size, buf_base, buf_offset)
        self._backend.synchronize_for_stream(raw_stream)
        self._ensure_extents_initialized(file_offset, size)
        lib = self._backend._library()
        done_total = 0
        cursor = buf_offset
        for device_off, seg_len, _logical in self._write_segments(file_offset, size):
            done = lib.write(
                self._dev_handle,
                buf_base,
                seg_len,
                device_off,
                cursor,
            )
            if done != seg_len:
                raise RuntimeError(
                    f"NDS write short I/O: got {done} bytes, expected {seg_len}"
                )
            done_total += seg_len
            cursor += seg_len
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        sub.result.value = done_total
        return sub

    def _run_sync_io(
        self,
        operation: str,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        self._backend._validate_stream(raw_stream)
        self._backend._validate_io_bounds(file_offset, size, buf_base, buf_offset)
        self._backend.synchronize_for_stream(raw_stream)
        lib = self._backend._library()
        if operation == "read":
            done = lib.read(
                self._handle, buf_base, size, file_offset, buf_offset
            )
        else:
            done = lib.write(
                self._handle, buf_base, size, file_offset, buf_offset
            )
        if done != size:
            raise RuntimeError(
                f"NDS {operation} short I/O: got {done} bytes, expected {size}"
            )
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        sub.result.value = size
        return sub
