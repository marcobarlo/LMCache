# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable coverage for the torch-fallback Ascend (B-lite) additions.

Three groups, all runnable without Ascend hardware:

- ``_tensor_from_ptr`` NPU dispatch: routing and storage construction are
  exercised with the torch internals monkeypatched (mirrors the MUSA
  pointer-view tests in ``tests/v1/test_torch_ops.py``).
- The ``NL_X_TWO_X_NB_BS_HS`` (MLA/DSA plane-tuple) branch of
  ``multi_layer_block_kv_transfer``: byte-exact D2H/H2D roundtrips on CPU
  tensors through the ``DeviceOps`` facade.
- The ``planes_per_layer`` layout hint: regrouping of a flat plane list
  into per-layer tuples inside ``normalize_and_discover_per_layer_formats``.
"""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.platform import resolve_device_ops
from lmcache.v1.platform import torch_ops as _py_ops
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

#: None of these tests allocate through the shared allocator, and the
#: autouse fixture's 5GB pinned allocation fails on hosts where the
#: lmcache_ascend plugin is installed but no ACL context can be established.
pytestmark = pytest.mark.no_shared_allocator

F = lmcache_native.EngineKVFormat


# ====================================================================== #
#  _tensor_from_ptr: NPU dispatch                                        #
# ====================================================================== #


def test_tensor_from_ptr_routes_npu_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NPU pointers are routed through the NPU pointer helper."""

    class FakeDevice:
        """Minimal fake device type for hosts without TorchNPU installed."""

        def __init__(self, value: object) -> None:
            self.type = str(value).split(":", maxsplit=1)[0]

    captured: dict[str, object] = {}

    def fake_npu_ptr(
        ptr: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: Any,
        total_bytes: int,
    ) -> torch.Tensor:
        captured.update(
            ptr=ptr,
            shape=shape,
            dtype=dtype,
            device_type=device.type,
            total_bytes=total_bytes,
        )
        return torch.empty(shape, dtype=dtype)

    monkeypatch.setattr(_py_ops.torch, "device", FakeDevice)
    monkeypatch.setattr(_py_ops, "_tensor_from_npu_ptr", fake_npu_ptr)

    tensor = _py_ops._tensor_from_ptr(0x1000, (2, 3), torch.float16, "npu:0")

    assert tensor.shape == (2, 3)
    assert captured == {
        "ptr": 0x1000,
        "shape": (2, 3),
        "dtype": torch.float16,
        "device_type": "npu",
        "total_bytes": 12,
    }


def test_tensor_from_npu_ptr_uses_external_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NPU pointer reconstruction returns a non-owning storage view."""

    fake_device = object()
    captured: dict[str, object] = {}

    class FakeStorage:
        def __init__(self, device: object) -> None:
            self.device = device

    class FakeTensor:
        def set_(
            self,
            storage: object,
            offset: int,
            shape: tuple[int, ...],
            stride: tuple[int, ...],
        ) -> None:
            captured["storage"] = storage
            captured["offset"] = offset
            captured["shape"] = shape
            captured["stride"] = stride

    fake_storage = FakeStorage(fake_device)
    fake_tensor = FakeTensor()

    def fake_construct_storage(ptr: int, device: object, total_bytes: int) -> object:
        captured["ptr"] = ptr
        captured["device"] = device
        captured["total_bytes"] = total_bytes
        return fake_storage

    def fake_empty(
        size: int,
        *,
        dtype: torch.dtype,
        device: object,
    ) -> FakeTensor:
        captured["empty_size"] = size
        captured["dtype"] = dtype
        captured["empty_device"] = device
        return fake_tensor

    monkeypatch.setattr(
        _py_ops.torch._C,
        "_construct_storage_from_data_pointer",
        fake_construct_storage,
        raising=False,
    )
    monkeypatch.setattr(_py_ops.torch, "empty", fake_empty)

    result = _py_ops._tensor_from_npu_ptr(
        0x1000, (2, 3), torch.float16, fake_device, 12
    )

    assert result is fake_tensor
    assert captured == {
        "ptr": 0x1000,
        "device": fake_device,
        "total_bytes": 12,
        "empty_size": 0,
        "dtype": torch.float16,
        "empty_device": fake_device,
        "storage": fake_storage,
        "offset": 0,
        "shape": (2, 3),
        "stride": (3, 1),
    }


def test_tensor_from_npu_ptr_fails_without_external_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NPU pointer reconstruction fails instead of returning a copy."""

    fake_device = object()

    def fake_construct_storage(_ptr: int, _device: object, _total_bytes: int) -> object:
        raise RuntimeError("storage construction unavailable")

    monkeypatch.setattr(
        _py_ops.torch._C,
        "_construct_storage_from_data_pointer",
        fake_construct_storage,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="failed to construct"):
        _py_ops._tensor_from_npu_ptr(0x1000, (2, 3), torch.float16, fake_device, 12)


# ====================================================================== #
#  multi_layer_block_kv_transfer: MLA/DSA plane-tuple format              #
# ====================================================================== #


def _make_tuple_planes(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    widths: tuple[int, ...],
) -> list[tuple[torch.Tensor, ...]]:
    """Build per-layer plane tuples with a distinct value per element.

    Every element encodes its (layer, plane) origin in its high digits so a
    misrouted slab fails loudly instead of coincidentally matching.
    """
    planes: list[tuple[torch.Tensor, ...]] = []
    for layer_idx in range(num_layers):
        layer = []
        for plane_idx, width in enumerate(widths):
            base = (layer_idx * len(widths) + plane_idx) * 100_000
            data = base + torch.arange(
                num_blocks * block_size * width, dtype=torch.float32
            )
            layer.append(data.reshape(num_blocks, block_size, 1, width))
        planes.append(tuple(layer))
    return planes


def _expected_d2h_chunks(
    planes: list[tuple[torch.Tensor, ...]],
    block_ids: list[int],
    chunk_tokens: int,
    blocks_per_object: int,
    block_size: int,
    skip_prefix_n_blocks: int,
) -> torch.Tensor:
    """Compute the expected staging chunks straight from the format geometry.

    ``[num_objects, L, chunk_tokens, sum(W_p)]``; token rows follow the
    block-id order, plane slabs are concatenated along the last axis, and
    regions outside the valid (skip-adjusted) block ranges stay zero.
    """
    num_layers = len(planes)
    widths = [int(p.shape[-1]) for p in planes[0]]
    num_objects = (len(block_ids) + blocks_per_object - 1) // blocks_per_object
    expected = torch.zeros(
        num_objects, num_layers, chunk_tokens, sum(widths), dtype=torch.float32
    )
    for object_idx in range(num_objects):
        flat_start = max(object_idx * blocks_per_object, skip_prefix_n_blocks)
        flat_end = min(
            object_idx * blocks_per_object + blocks_per_object, len(block_ids)
        )
        token = (flat_start - object_idx * blocks_per_object) * block_size
        for flat_idx in range(flat_start, flat_end):
            for layer_idx, layer_planes in enumerate(planes):
                slab = 0
                for plane, width in zip(layer_planes, widths, strict=True):
                    block = plane[block_ids[flat_idx]].reshape(block_size, width)
                    dst = expected[object_idx, layer_idx, token : token + block_size]
                    dst[:, slab : slab + width] = block
                    slab += width
            token += block_size
    return expected


def _expected_h2d_planes(
    planes: list[tuple[torch.Tensor, ...]],
    block_ids: list[int],
    skip_prefix_n_blocks: int,
) -> list[tuple[torch.Tensor, ...]]:
    """Compute the expected paged planes after an H2D transfer.

    Blocks referenced by a valid (skip-adjusted) flat position receive their
    original data back; every other block stays zero.
    """
    expected = [tuple(torch.zeros_like(plane) for plane in layer) for layer in planes]
    # The per-object valid ranges tile [skip, n_blocks) contiguously, so the
    # transferred flat positions are exactly the non-skipped block ids.
    for flat_idx in range(skip_prefix_n_blocks, len(block_ids)):
        for layer_idx, layer_planes in enumerate(planes):
            for plane_idx, plane in enumerate(layer_planes):
                expected[layer_idx][plane_idx][block_ids[flat_idx]] = plane[
                    block_ids[flat_idx]
                ]
    return expected


@pytest.mark.parametrize("widths", [(6, 2), (6, 2, 4)], ids=["mla", "dsa"])
@pytest.mark.parametrize("skip", [0, 1], ids=["no_skip", "skip_one"])
def test_mla_tuple_block_kv_transfer_roundtrip(
    widths: tuple[int, ...], skip: int
) -> None:
    """D2H then H2D through the DeviceOps facade is byte-exact both ways."""
    num_layers, num_blocks, block_size = 3, 5, 4
    chunk_tokens = 8
    blocks_per_object = chunk_tokens // block_size
    block_ids = [3, 1, 4, 0]

    ops = resolve_device_ops("cpu")
    planes = _make_tuple_planes(num_layers, num_blocks, block_size, widths)
    width_sum = sum(widths)

    shape_desc = PageBufferShapeDesc()
    shape_desc.nl = num_layers
    shape_desc.nb = num_blocks
    shape_desc.bs = block_size
    shape_desc.nh = 1
    shape_desc.hs = width_sum
    shape_desc.kv_size = 1
    shape_desc.element_size = 4
    shape_desc.dtype = torch.float32

    chunks = [
        torch.zeros(num_layers, chunk_tokens, width_sum, dtype=torch.float32)
        for _ in range(len(block_ids) // blocks_per_object)
    ]
    ops.multi_layer_block_kv_transfer(
        planes,
        chunks,
        block_ids,
        "cpu",
        lmcache_native.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        F.NL_X_TWO_X_NB_BS_HS,
        skip,
    )
    expected_chunks = _expected_d2h_chunks(
        planes, block_ids, chunk_tokens, blocks_per_object, block_size, skip
    )
    for object_idx, chunk in enumerate(chunks):
        assert torch.equal(chunk, expected_chunks[object_idx]), (
            f"D2H chunk {object_idx} mismatch (widths={widths}, skip={skip})"
        )

    target_planes = [
        tuple(torch.zeros_like(plane) for plane in layer) for layer in planes
    ]
    ops.multi_layer_block_kv_transfer(
        target_planes,
        chunks,
        block_ids,
        "cpu",
        lmcache_native.TransferDirection.H2D,
        shape_desc,
        chunk_tokens,
        F.NL_X_TWO_X_NB_BS_HS,
        skip,
    )
    expected_planes = _expected_h2d_planes(planes, block_ids, skip)
    for layer_idx, (got, exp) in enumerate(
        zip(target_planes, expected_planes, strict=True)
    ):
        for plane_idx, (got_plane, exp_plane) in enumerate(zip(got, exp, strict=True)):
            assert torch.equal(got_plane, exp_plane), (
                f"H2D plane mismatch layer={layer_idx} plane={plane_idx} "
                f"(widths={widths}, skip={skip})"
            )


# ====================================================================== #
#  _normalize_lmcache_objects: pointer mode follows the transfer device   #
# ====================================================================== #


def _mla_shape_desc(
    num_layers: int, chunk_tokens: int, width_sum: int
) -> PageBufferShapeDesc:
    """Minimal MLA-family shape desc for pointer-mode reconstruction."""
    shape_desc = PageBufferShapeDesc()
    shape_desc.nl = num_layers
    shape_desc.nb = 1
    shape_desc.bs = chunk_tokens
    shape_desc.nh = 1
    shape_desc.hs = width_sum
    shape_desc.kv_size = 1
    shape_desc.element_size = 4
    shape_desc.dtype = torch.float32
    return shape_desc


@pytest.mark.parametrize("device", [None, "cpu"], ids=["default", "explicit_cpu"])
def test_normalize_lmcache_objects_pointer_mode_stays_cpu(device: object) -> None:
    """Pointer mode with no/explicit CPU device keeps aliasing CPU memory."""
    chunk = torch.zeros(3, 8, 6, dtype=torch.float32)
    tensors = _py_ops._normalize_lmcache_objects(
        [chunk.data_ptr()],
        shape_desc=_mla_shape_desc(3, 8, 6),
        lmcache_chunk_size=8,
        engine_kv_format=F.NL_X_NB_BS_HS,
        dtype=torch.float32,
        device=device,
    )
    assert len(tensors) == 1
    assert tensors[0].device.type == "cpu"
    assert tuple(tensors[0].shape) == (3, 8, 6)
    # Byte-identical aliasing: writes through the view reach the original.
    tensors[0][0, 0, 0] = 7.0
    assert chunk[0, 0, 0] == 7.0


def test_normalize_lmcache_objects_pointer_mode_honors_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pointer mode reconstructs on the caller's device, not hard-coded CPU."""

    class FakeDevice:
        def __init__(self, value: object) -> None:
            self.type = str(value).split(":", maxsplit=1)[0]

    captured: dict[str, object] = {}

    def fake_npu_ptr(
        ptr: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: Any,
        total_bytes: int,
    ) -> torch.Tensor:
        captured.update(
            ptr=ptr,
            shape=shape,
            dtype=dtype,
            device_type=device.type,
            total_bytes=total_bytes,
        )
        return torch.empty(shape, dtype=dtype)

    monkeypatch.setattr(_py_ops.torch, "device", FakeDevice)
    monkeypatch.setattr(_py_ops, "_tensor_from_npu_ptr", fake_npu_ptr)

    buffer = torch.zeros(3 * 8 * 6, dtype=torch.float32)
    tensors = _py_ops._normalize_lmcache_objects(
        [buffer.data_ptr()],
        shape_desc=_mla_shape_desc(3, 8, 6),
        lmcache_chunk_size=8,
        engine_kv_format=F.NL_X_NB_BS_HS,
        dtype=torch.float32,
        device="npu:0",
    )

    assert captured == {
        "ptr": buffer.data_ptr(),
        "shape": (3, 8, 6),
        "dtype": torch.float32,
        "device_type": "npu",
        "total_bytes": 3 * 8 * 6 * 4,
    }
    assert len(tensors) == 1


def test_block_transfer_pointer_mode_objects_roundtrip_cpu() -> None:
    """Through the facade, pointer-mode CPU objects stay byte-identical."""
    num_layers, num_blocks, block_size = 3, 5, 4
    chunk_tokens = 8
    blocks_per_object = chunk_tokens // block_size
    block_ids = [3, 1, 4, 0]
    widths = (6, 2)

    ops = resolve_device_ops("cpu")
    planes = _make_tuple_planes(num_layers, num_blocks, block_size, widths)
    width_sum = sum(widths)
    shape_desc = _mla_shape_desc(num_layers, chunk_tokens, width_sum)
    shape_desc.nb = num_blocks
    shape_desc.bs = block_size

    chunks = [
        torch.zeros(num_layers, chunk_tokens, width_sum, dtype=torch.float32)
        for _ in range(len(block_ids) // blocks_per_object)
    ]
    ops.multi_layer_block_kv_transfer(
        planes,
        [chunk.data_ptr() for chunk in chunks],
        block_ids,
        "cpu",
        lmcache_native.TransferDirection.D2H,
        shape_desc,
        chunk_tokens,
        F.NL_X_TWO_X_NB_BS_HS,
        0,
    )
    expected_chunks = _expected_d2h_chunks(
        planes, block_ids, chunk_tokens, blocks_per_object, block_size, 0
    )
    for object_idx, chunk in enumerate(chunks):
        assert torch.equal(chunk, expected_chunks[object_idx]), (
            f"pointer-mode D2H chunk {object_idx} differs from tensor mode"
        )


# ====================================================================== #
#  normalize_and_discover_per_layer_formats: planes_per_layer regroup     #
# ====================================================================== #

NB, NL, BS, HS = 7, 5, 3, 4


def _plane(width: int) -> torch.Tensor:
    """One paged plane ``[NB, BS, 1, width]`` as registered by vLLM-Ascend."""
    return torch.zeros((NB, BS, 1, width), dtype=torch.float16)


def _flat_planes(widths: tuple[int, ...]) -> list[torch.Tensor]:
    """A flat per-plane registration list: ``NL`` consecutive plane groups."""
    flat: list[torch.Tensor] = []
    for _ in range(NL):
        flat.extend(_plane(width) for width in widths)
    return flat


def test_regroup_two_planes_detects_mla_tuple() -> None:
    """A flat 2*NL latent+rope list regroups into the MLA tuple format."""
    flat = _flat_planes((HS * 8, HS))
    normalized, formats = normalize_and_discover_per_layer_formats(
        flat,
        [],
        EngineType.VLLM,
        {"kv_layout": "NHD", "planes_per_layer": 2},
    )
    assert formats == [F.NL_X_TWO_X_NB_BS_HS] * NL
    assert len(normalized) == NL
    # The regrouped planes must alias the registered buffers, not copy them.
    for layer_idx, layer in enumerate(normalized):
        assert isinstance(layer, (list, tuple))
        assert len(layer) == 2
        flat_idx = layer_idx * 2
        assert [p.data_ptr() for p in layer] == [
            flat[flat_idx].data_ptr(),
            flat[flat_idx + 1].data_ptr(),
        ]


def test_regroup_three_planes_detects_dsa_tuple() -> None:
    """A flat 3*NL latent+rope+dsa list regroups into the same tuple format."""
    normalized, formats = normalize_and_discover_per_layer_formats(
        _flat_planes((HS * 8, HS, HS * 2)),
        [],
        EngineType.VLLM,
        {"kv_layout": "NHD", "planes_per_layer": 3},
    )
    assert formats == [F.NL_X_TWO_X_NB_BS_HS] * NL
    assert len(normalized) == NL
    assert all(len(layer) == 3 for layer in normalized)


def test_regroup_default_one_keeps_today_classification() -> None:
    """Without the hint a flat plane list keeps its pre-existing format."""
    normalized, formats = normalize_and_discover_per_layer_formats(
        _flat_planes((HS * 8, HS)),
        [],
        EngineType.VLLM,
        {"kv_layout": "NHD"},
    )
    assert formats == [F.NL_X_NB_BS_NH_CS] * (2 * NL)
    assert len(normalized) == 2 * NL


def test_regroup_skips_already_grouped_tuples() -> None:
    """Already-tuple input is left alone even when the hint is present."""
    tuples = [(_plane(HS * 8), _plane(HS)) for _ in range(NL)]
    normalized, formats = normalize_and_discover_per_layer_formats(
        tuples,
        [],
        EngineType.VLLM,
        {"kv_layout": "NHD", "planes_per_layer": 2},
    )
    assert formats == [F.NL_X_TWO_X_NB_BS_HS] * NL
    assert len(normalized) == NL


def test_regroup_rejects_indivisible_plane_count() -> None:
    """A flat list whose length is not a multiple of the hint raises."""
    flat = _flat_planes((HS * 8, HS))[: 2 * NL - 1]
    with pytest.raises(ValueError, match="planes_per_layer"):
        normalize_and_discover_per_layer_formats(
            flat,
            [],
            EngineType.VLLM,
            {"kv_layout": "NHD", "planes_per_layer": 2},
        )
