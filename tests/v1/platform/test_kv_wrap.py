# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for KV-cache wrapping of per-layer tuple values."""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import kv_wrap

pytestmark = pytest.mark.no_shared_allocator


class _RecordingFactory:
    def __init__(self) -> None:
        self.wrapped: list[torch.Tensor] = []

    def __call__(self, tensor: torch.Tensor) -> Any:
        self.wrapped.append(tensor)
        return f"wrapper-{len(self.wrapped)}"


def test_flatten_expands_per_layer_tuples_in_order() -> None:
    k0, v0 = torch.zeros(2), torch.zeros(3)
    k1, v1 = torch.zeros(4), torch.zeros(5)
    kv_caches = {
        "layer.0": (k0, v0),
        "layer.1": (k1, v1),
    }

    flat = kv_wrap.flatten_kv_cache_values(kv_caches)

    assert flat == [k0, v0, k1, v1]


def test_flatten_passes_plain_tensors_through() -> None:
    t0, t1 = torch.zeros(2), torch.zeros(3)

    flat = kv_wrap.flatten_kv_cache_values({"a": t0, "b": t1})

    assert flat == [t0, t1]


def test_planes_per_layer_reads_uniform_tuple_arity() -> None:
    assert kv_wrap.planes_per_layer({"l": (torch.zeros(1), torch.zeros(1))}) == 2
    assert (
        kv_wrap.planes_per_layer(
            {"l": (torch.zeros(1), torch.zeros(1), torch.zeros(1))}
        )
        == 3
    )


def test_planes_per_layer_defaults_for_flat_or_mixed_values() -> None:
    assert kv_wrap.planes_per_layer({"l": torch.zeros(1)}) == 1
    assert kv_wrap.planes_per_layer({}) == 1
    # Mixed tuple / tensor values carry no single arity; keep the default so
    # the server-side detection surfaces the inconsistency loudly.
    assert (
        kv_wrap.planes_per_layer(
            {"a": (torch.zeros(1), torch.zeros(1)), "b": torch.zeros(1)}
        )
        == 1
    )
    assert (
        kv_wrap.planes_per_layer(
            {"a": (torch.zeros(1),), "b": (torch.zeros(1), torch.zeros(1))}
        )
        == 1
    )


def test_wrap_kv_caches_wraps_flattened_tuple_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _RecordingFactory()
    monkeypatch.setattr(kv_wrap, "wrap_one_kv_cache", factory)
    k0, v0, k1, v1 = (torch.zeros(i + 1) for i in range(4))

    wrappers = kv_wrap.wrap_kv_caches({"layer.0": (k0, v0), "layer.1": (k1, v1)})

    assert factory.wrapped == [k0, v0, k1, v1]
    assert wrappers == ["wrapper-1", "wrapper-2", "wrapper-3", "wrapper-4"]
