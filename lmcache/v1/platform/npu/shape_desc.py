# SPDX-License-Identifier: Apache-2.0
"""NPU ``PageBufferShapeDesc``: native fields plus fmt-17 plane extras.

CUDA's compiled struct is unchanged. These attributes live on a Python
subclass of ``lmcache_native.PageBufferShapeDesc`` (``dynamic_attr``).
Ascend C++ duck-copies them off a generic ``py::object``.
"""

from __future__ import annotations

import lmcache.lmcache_native as lmcache_native


class NpuPageBufferShapeDesc(lmcache_native.PageBufferShapeDesc):
    """Native shape desc plus per-plane slot widths for packed MLA.

    ``num_planes == 0`` and empty ``plane_slot_bytes`` mean unset: the
    Ascend host must not infer packed K/V from ``hs % 32``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.num_planes: int = 0
        self.plane_slot_bytes: tuple[int, ...] = ()
        self.plane_block_stride_bytes: tuple[int, ...] = ()
