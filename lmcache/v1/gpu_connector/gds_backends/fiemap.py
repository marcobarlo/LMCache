# SPDX-License-Identifier: Apache-2.0
"""``FS_IOC_FIEMAP`` query for a file range.

The ioctl buffer is one ``struct fiemap`` header, then
``_FIEMAP_EXTENT_BATCH`` ``struct fiemap_extent`` records. Little-endian.

struct fiemap:
  offset  0  u64  fm_start
  offset  8  u64  fm_length
  offset 16  u32  fm_flags
  offset 20  u32  fm_mapped_extents
  offset 24  u32  fm_extent_count
  offset 28  u32  fm_reserved
  offset 32  struct fiemap_extent fm_extents[]

struct fiemap_extent:
  offset  0  u64  fe_logical
  offset  8  u64  fe_physical
  offset 16  u64  fe_length
  offset 24  u64  fe_reserved64[2]
  offset 40  u32  fe_flags
  offset 44  u32  fe_reserved[3]
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
import fcntl
import struct

FS_IOC_FIEMAP = 0xC020660B

_FIEMAP_U64 = 8
_FIEMAP_U32 = 4
_FE_RESERVED64_COUNT = 2
_FE_RESERVED32_COUNT = 3

_FM_START = 0
_FM_LENGTH = _FM_START + _FIEMAP_U64
_FM_FLAGS = _FM_LENGTH + _FIEMAP_U64
_FM_MAPPED_EXTENTS = _FM_FLAGS + _FIEMAP_U32
_FM_EXTENT_COUNT = _FM_MAPPED_EXTENTS + _FIEMAP_U32
_FM_RESERVED = _FM_EXTENT_COUNT + _FIEMAP_U32
_FIEMAP_HEADER_SIZE = _FM_RESERVED + _FIEMAP_U32

_FE_LOGICAL = 0
_FE_PHYSICAL = _FE_LOGICAL + _FIEMAP_U64
_FE_LENGTH = _FE_PHYSICAL + _FIEMAP_U64
_FE_RESERVED64 = _FE_LENGTH + _FIEMAP_U64
_FE_FLAGS = _FE_RESERVED64 + _FE_RESERVED64_COUNT * _FIEMAP_U64
_FE_RESERVED = _FE_FLAGS + _FIEMAP_U32
_FIEMAP_EXTENT_SIZE = _FE_RESERVED + _FE_RESERVED32_COUNT * _FIEMAP_U32

_FIEMAP_HEADER_FMT = "<QQIIII"
_FIEMAP_EXTENT_RANGE_FMT = "<QQQ"
_FIEMAP_EXTENT_FLAGS_FMT = "<I"
# One call covers a 128–256 GiB fallocated slab (~2 GiB extents on this ext4).
_FIEMAP_EXTENT_BATCH = 256

FIEMAP_FLAG_SYNC = 0x1
FIEMAP_EXTENT_LAST = 0x00000001
FIEMAP_EXTENT_UNKNOWN = 0x00000002
FIEMAP_EXTENT_DELALLOC = 0x00000004
FIEMAP_EXTENT_ENCODED = 0x00000008
FIEMAP_EXTENT_DATA_ENCRYPTED = 0x00000080
FIEMAP_EXTENT_NOT_ALIGNED = 0x00000100
FIEMAP_EXTENT_DATA_INLINE = 0x00000200
FIEMAP_EXTENT_DATA_TAIL = 0x00000400
FIEMAP_EXTENT_UNWRITTEN = 0x00000800

# NDS store cannot DMA these. UNWRITTEN is allowed and zero-filled first.
_UNSUPPORTED_FIEMAP_FLAGS = (
    FIEMAP_EXTENT_UNKNOWN
    | FIEMAP_EXTENT_DELALLOC
    | FIEMAP_EXTENT_ENCODED
    | FIEMAP_EXTENT_DATA_ENCRYPTED
    | FIEMAP_EXTENT_NOT_ALIGNED
    | FIEMAP_EXTENT_DATA_INLINE
    | FIEMAP_EXTENT_DATA_TAIL
)
_EXTENT_SECTOR = 512


@dataclass(frozen=True)
class FileExtent:
    """One file extent: logical byte range and the disk physical range."""

    logical: int
    physical: int
    length: int
    flags: int

    @property
    def unwritten(self) -> bool:
        return bool(self.flags & FIEMAP_EXTENT_UNWRITTEN)


def fiemap_extents(fd: int, size: int) -> list[FileExtent]:
    """Query ``FS_IOC_FIEMAP`` for ``[0, size)`` on an open file fd.

    Returns extents trimmed to ``size``, in ascending logical order, until
    ``FIEMAP_EXTENT_LAST`` or the range is exhausted. A record that contributes
    no bytes inside ``[0, size)`` (zero length, or entirely outside the range)
    is skipped. This function does not raise for that.
    :meth:`FileExtentMap.load` validates that the records which remain cover
    ``[0, size)`` without gaps.
    """
    fm_start = 0
    out: list[FileExtent] = []
    while fm_start < size:
        buf = bytearray(
            _FIEMAP_HEADER_SIZE + _FIEMAP_EXTENT_BATCH * _FIEMAP_EXTENT_SIZE
        )
        query_len = size - fm_start
        struct.pack_into(
            _FIEMAP_HEADER_FMT,
            buf,
            _FM_START,
            fm_start,  # fm_start
            query_len,  # fm_length
            FIEMAP_FLAG_SYNC,  # fm_flags
            0,  # fm_mapped_extents (kernel output)
            _FIEMAP_EXTENT_BATCH,  # fm_extent_count
            0,  # fm_reserved
        )
        fcntl.ioctl(fd, FS_IOC_FIEMAP, buf, True)
        _, _, _, mapped, _, _ = struct.unpack_from(_FIEMAP_HEADER_FMT, buf, _FM_START)
        if mapped == 0:
            break
        mapped = min(mapped, _FIEMAP_EXTENT_BATCH)
        last_logical = fm_start
        for i in range(mapped):
            base = _FIEMAP_HEADER_SIZE + i * _FIEMAP_EXTENT_SIZE
            logical, physical, length = struct.unpack_from(
                _FIEMAP_EXTENT_RANGE_FMT, buf, base + _FE_LOGICAL
            )
            (flags,) = struct.unpack_from(
                _FIEMAP_EXTENT_FLAGS_FMT, buf, base + _FE_FLAGS
            )
            # No bytes inside [0, size). Skip; the caller checks coverage.
            if logical + length <= 0 or length == 0:
                continue
            if logical >= size:
                continue
            trim_end = min(logical + length, size)
            trim_start = max(logical, 0)
            if trim_end <= trim_start:
                continue
            out.append(
                FileExtent(
                    logical=trim_start,
                    physical=physical + (trim_start - logical),
                    length=trim_end - trim_start,
                    flags=flags,
                )
            )
            last_logical = max(last_logical, logical + length)
            if flags & FIEMAP_EXTENT_LAST:
                if last_logical >= size:
                    return out
                break
        if last_logical <= fm_start:
            break
        fm_start = last_logical
    return out


def validate_slab_extents(extents: Sequence[FileExtent], size: int) -> None:
    """Reject slabs NDS store cannot use: gaps, short coverage, alignment, bad flags."""
    if not extents:
        raise ValueError("NDS slab has no FIEMAP extents")
    pos = 0
    for ext in extents:
        if ext.logical != pos:
            raise ValueError(
                f"NDS slab extents have a gap at byte {pos} (got logical {ext.logical})"
            )
        if ext.length <= 0 or ext.length % _EXTENT_SECTOR:
            raise ValueError(
                "NDS extent length must be a positive multiple of "
                f"{_EXTENT_SECTOR}: {ext}"
            )
        if ext.physical <= 0 or ext.physical % _EXTENT_SECTOR:
            raise ValueError(
                "NDS extent physical offset must be a positive multiple of "
                f"{_EXTENT_SECTOR}: {ext}"
            )
        bad = ext.flags & _UNSUPPORTED_FIEMAP_FLAGS
        if bad:
            raise ValueError(
                f"NDS extent has unsupported FIEMAP flags 0x{bad:x}: {ext}"
            )
        pos += ext.length
    if pos < size:
        raise ValueError(f"NDS slab extents cover {pos} bytes, need {size}")


@dataclass(frozen=True)
class FileExtentMap:
    """Validated FIEMAP coverage of ``[0, size)`` for an NDS store slab."""

    extents: tuple[FileExtent, ...]
    size: int

    def __post_init__(self) -> None:
        validate_slab_extents(self.extents, self.size)

    @classmethod
    def load(cls, fd: int, size: int) -> FileExtentMap:
        """Query ``FS_IOC_FIEMAP`` and reject a map NDS store cannot use."""
        return cls(tuple(fiemap_extents(fd, size)), size)

    def extent_at(self, logical_offset: int) -> FileExtent:
        for ext in self.extents:
            if ext.logical <= logical_offset < ext.logical + ext.length:
                return ext
        raise RuntimeError(
            f"NDS slab logical offset {logical_offset} is not covered by FIEMAP"
        )
