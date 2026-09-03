# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the NPU event IPC backend (no Ascend hardware)."""

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.platform.base.event_ipc import EventIPCBackend
from lmcache.v1.platform.npu import NpuDeviceSpec

pytestmark = pytest.mark.no_shared_allocator


class _FakeEvent:
    """torch.npu-style Event with the CUDA interprocess ABI."""

    def __init__(self, interprocess: bool = False) -> None:
        if not interprocess:
            raise ValueError("fake events must be interprocess")

    @classmethod
    def from_ipc_handle(cls, device: Any, handle: bytes) -> "_FakeEvent":
        return cls(interprocess=True)

    def ipc_handle(self) -> bytes:
        return b"npu-handle"

    def record(self, stream: Any = None) -> None:
        pass

    def wait(self, stream: Any = None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


class _FakeNpuModule:
    Event = _FakeEvent


class _AbiLessModule:
    class Event:  # noqa: N801 - mirrors torch module surface
        pass


def _device() -> Any:
    class _Device:
        type = "npu"

    return _Device()


def test_backend_satisfies_protocol() -> None:
    from lmcache.v1.platform.npu.event_ipc import NpuEventIPCBackend

    backend = NpuEventIPCBackend(event_module=_FakeNpuModule())
    assert isinstance(backend, EventIPCBackend)
    assert backend.device_type == "npu"


def test_check_event_support_fails_closed_without_abi() -> None:
    from lmcache.v1.platform.npu.event_ipc import NpuEventIPCBackend

    backend = NpuEventIPCBackend(event_module=_AbiLessModule())
    with pytest.raises(RuntimeError, match="interprocess"):
        backend.check_event_support(_device())


def test_check_event_support_passes_with_abi() -> None:
    from lmcache.v1.platform.npu.event_ipc import NpuEventIPCBackend

    backend = NpuEventIPCBackend(event_module=_FakeNpuModule())
    backend.check_event_support(_device())


def test_backend_creates_exports_and_imports() -> None:
    from lmcache.v1.platform.npu.event_ipc import NpuEventIPCBackend

    backend = NpuEventIPCBackend(event_module=_FakeNpuModule())
    event = backend.create_event(_device())
    handle = backend.export_event(event, _device())
    assert handle == b"npu-handle"
    imported = backend.import_event(handle, _device())
    assert isinstance(imported, _FakeEvent)


def test_torch_npu_module_raises_without_torch_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from lmcache.v1.platform.npu import event_ipc

    monkeypatch.delattr(torch, "npu", raising=False)
    with pytest.raises(RuntimeError, match="torch_npu"):
        event_ipc._torch_npu_module()


def test_device_spec_exposes_cached_event_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lmcache.v1.platform.npu import event_ipc

    monkeypatch.setattr(event_ipc, "_torch_npu_module", lambda: _FakeNpuModule())
    spec = NpuDeviceSpec()
    first = spec.event_ipc_backend
    assert first.device_type == "npu"
    assert spec.event_ipc_backend is first


def test_export_pins_source_events_in_bounded_cache() -> None:
    """CANN invalidates a handle once its source event is destroyed."""
    from lmcache.v1.platform.npu.event_ipc import (
        _EXPORT_LIVENESS_CACHE,
        NpuEventIPCBackend,
    )

    backend = NpuEventIPCBackend(event_module=_FakeNpuModule())
    events = [backend.create_event(_device()) for _ in range(3)]
    for event in events:
        backend.export_event(event, _device())

    assert list(backend._exported_events) == events

    for _ in range(_EXPORT_LIVENESS_CACHE):
        backend.export_event(backend.create_event(_device()), _device())
    assert len(backend._exported_events) == _EXPORT_LIVENESS_CACHE
    assert events[0] not in backend._exported_events
