# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for NPU stream-ordered completion recording."""

# Third Party
import pytest

# First Party
from lmcache.v1.platform.npu import device_ops as npu_device_ops_module
from lmcache.v1.platform.npu.device_ops import NpuDeviceOps

pytestmark = pytest.mark.no_shared_allocator


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
    assert events[0][0] == "mp.store.start"
    assert events[0][1] == "session"


def test_sync_helper_raises_on_acl_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingRt:
        @staticmethod
        def synchronize_stream(ptr: int) -> int:
            return 507899

    import types

    fake_acl = types.ModuleType("acl")
    fake_acl.rt = _FailingRt
    monkeypatch.setitem(__import__("sys").modules, "acl", fake_acl)
    with pytest.raises(RuntimeError, match="507899"):
        npu_device_ops_module._synchronize_npu_stream_pointer(7)
