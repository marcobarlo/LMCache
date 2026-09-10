# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for NPU stream-ordered completion recording."""

# Standard
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.npu import device_ops as npu_device_ops_module
from lmcache.v1.platform.npu.device_ops import NpuDeviceOps

pytestmark = pytest.mark.no_shared_allocator


def _install_fake_c_ops(
    monkeypatch: pytest.MonkeyPatch, **symbols: object
) -> None:
    fake_c_ops = types.ModuleType("lmcache_ascend.c_ops")
    for name, value in symbols.items():
        setattr(fake_c_ops, name, value)
    fake_pkg = types.ModuleType("lmcache_ascend")
    fake_pkg.c_ops = fake_c_ops
    monkeypatch.setitem(sys.modules, "lmcache_ascend", fake_pkg)
    monkeypatch.setitem(sys.modules, "lmcache_ascend.c_ops", fake_c_ops)


def test_record_completion_syncs_stream_before_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = NpuDeviceOps()
    ops.drain_recorded_completions()  # isolate the shared fallback buffer
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        npu_device_ops_module,
        "_synchronize_npu_stream_pointer",
        lambda ptr: calls.append(("sync", ptr)),
    )

    ops.record_completion_on_stream(0xABCD, "finish_write", b"payload")

    assert calls == [("sync", 0xABCD)]
    assert ops.drain_recorded_completions() == [("finish_write", b"payload")]


def test_record_event_syncs_stream_before_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = NpuDeviceOps()
    ops.drain_recorded_events()
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        npu_device_ops_module,
        "_synchronize_npu_stream_pointer",
        lambda ptr: calls.append(("sync", ptr)),
    )

    ops.record_event_on_stream(
        0xBEEF, "mp.store.start", "session", {"device": "npu"}, {"engine_id": 1}
    )

    assert calls == [("sync", 0xBEEF)]
    events = ops.drain_recorded_events()
    assert len(events) == 1
    event_type, session_id, _timestamp, string_metadata, int_metadata = events[0]
    assert event_type == "mp.store.start"
    assert session_id == "session"
    assert string_metadata == {"device": "npu"}
    assert int_metadata == {"engine_id": 1}


def test_sync_helper_raises_on_acl_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingRt:
        @staticmethod
        def synchronize_stream(ptr: int) -> int:
            return 507899

    fake_acl = types.ModuleType("acl")
    fake_acl.rt = _FailingRt
    monkeypatch.setitem(sys.modules, "acl", fake_acl)
    with pytest.raises(RuntimeError, match="507899"):
        npu_device_ops_module._synchronize_npu_stream_pointer(7)


@pytest.mark.parametrize(
    "symbol",
    [
        "record_completion_on_stream",
        "record_event_on_stream",
        "lmcache_memcpy_async",
    ],
)
def test_c_ops_torch_reexports_are_not_instance_bound(
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    """Plugin re-exports of the torch fallback must not defeat the overrides.

    ``lmcache_ascend.__init__`` copies torch-fallback symbols the compiled
    c_ops module lacks onto c_ops; ``bind_native`` would then instance-bind
    them over the class-level sync-ordered overrides whenever ``lmcache``
    is imported before the plugin (the natural server-process order).
    """
    _install_fake_c_ops(monkeypatch, **{symbol: getattr(torch_ops, symbol)})

    ops = NpuDeviceOps()
    ops.ensure_native()
    assert symbol not in vars(ops)

    if symbol != "record_completion_on_stream":
        return
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        npu_device_ops_module,
        "_synchronize_npu_stream_pointer",
        lambda ptr: calls.append(("sync", ptr)),
    )
    ops.record_completion_on_stream(7, "finish_write", b"payload")
    assert calls == [("sync", 7)]


def test_genuine_native_recorder_binding_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real native recorder must survive the re-export cleanup."""

    def _native_recorder(stream_ptr: int, kind: str, payload: bytes) -> None:
        pass

    def _native_event_recorder(*args: object) -> None:
        pass

    _install_fake_c_ops(
        monkeypatch,
        record_completion_on_stream=_native_recorder,
        record_event_on_stream=_native_event_recorder,
    )

    ops = NpuDeviceOps()
    ops.ensure_native()
    assert ops.__dict__["record_completion_on_stream"] is _native_recorder
    assert ops.__dict__["record_event_on_stream"] is _native_event_recorder
