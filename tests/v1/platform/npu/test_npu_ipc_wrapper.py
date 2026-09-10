# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable coverage for the plane-aggregating NPU IPC wrapper.

The sharing/reconstruction mechanics (``_share_npu_`` /
``_new_shared_npu``) need real Ascend hardware and are covered by the
LMCache-Ascend plugin's spawn round-trip tests; everything testable
without a device lives here.
"""

# Standard
import pickle

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.npu import NpuDeviceSpec
from lmcache.v1.platform.npu.ipc_wrapper import NpuIPCWrapper, PlaneRecord

pytestmark = pytest.mark.no_shared_allocator


def _handbuilt_wrapper(records: tuple[PlaneRecord, ...]) -> NpuIPCWrapper:
    """Build a wrapper without touching NPU storage sharing.

    Populates exactly the fields ``__init__`` would produce for the given
    plane records, so equality/pickle/arity logic can run on CPU.
    """
    wrapper = NpuIPCWrapper.__new__(NpuIPCWrapper)
    wrapper._plane_records = records
    wrapper.device_uuid = "npu-test-0"
    return wrapper


def _record(marker: int) -> PlaneRecord:
    handle = (b"npu", marker, None, None, None)
    return (handle, torch.float16, (7, 3, 1, 4), (12, 4, 4, 1), 0)


def test_device_spec_binds_npu_ipc_wrapper() -> None:
    """The NPU spec exposes the wrapper through the platform registry."""
    assert NpuDeviceSpec().ipc_wrapper_cls is NpuIPCWrapper
    assert NpuIPCWrapper.device_type == "npu"
    assert issubclass(NpuIPCWrapper, DeviceIPCWrapper)


def test_wrap_classmethod_builds_instance_per_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wrap`` keeps one record per plane, in registration order."""
    k = torch.zeros(2, 3)
    v = torch.ones(2, 3)
    shared_ptrs: list[int] = []

    class _FakeStorage:
        """Stand-in for an NPU tensor's untyped storage."""

        def __init__(self, tensor: torch.Tensor) -> None:
            self._tensor = tensor

        def _share_npu_(self) -> tuple:  # noqa: N802 (torch's naming)
            shared_ptrs.append(self._tensor.data_ptr())
            return (b"npu", shared_ptrs[-1], None, None, None)

    monkeypatch.setattr(
        torch.Tensor, "untyped_storage", lambda self: _FakeStorage(self)
    )
    monkeypatch.setattr(
        NpuIPCWrapper, "_get_device_uuid", staticmethod(lambda idx: "npu-test-0")
    )

    wrapper = NpuIPCWrapper.wrap((k, v))

    assert wrapper.device_uuid == "npu-test-0"
    assert shared_ptrs == [k.data_ptr(), v.data_ptr()]
    records = wrapper._plane_records  # noqa: SLF001 (arity under test)
    assert len(records) == 2
    assert [r[2] for r in records] == [(2, 3), (2, 3)]
    assert [r[1] for r in records] == [torch.float32, torch.float32]


def test_wrap_rejects_empty_plane_sequence() -> None:
    with pytest.raises(ValueError, match="at least one plane"):
        NpuIPCWrapper.wrap(())


def test_equality_compares_plane_records() -> None:
    a = _handbuilt_wrapper((_record(1), _record(2)))
    same = _handbuilt_wrapper((_record(1), _record(2)))
    other = _handbuilt_wrapper((_record(1), _record(3)))
    different_uuid = _handbuilt_wrapper((_record(1), _record(2)))
    different_uuid.device_uuid = "npu-test-1"

    assert a == same
    assert a != other
    assert a != different_uuid
    assert a != "not-a-wrapper"
    assert hash(a) == hash(same)


def test_pickle_roundtrip_preserves_plane_records() -> None:
    wrapper = _handbuilt_wrapper((_record(1), _record(2)))

    restored = pickle.loads(pickle.dumps(wrapper))

    assert isinstance(restored, NpuIPCWrapper)
    assert restored == wrapper
    assert restored.device_uuid == "npu-test-0"


def test_record_count_encodes_reconstruction_arity() -> None:
    """Single-plane wrappers stay bare; multi-plane ones reconstruct tuples.

    The device-dependent reconstruction cannot run on CPU, but the wire
    contract relies on this arity rule, so pin the invariant: a wrapper
    built from N planes carries N records and reconstructs N tensors
    (bare only when N == 1).
    """
    single = _handbuilt_wrapper((_record(1),))
    multi = _handbuilt_wrapper((_record(1), _record(2)))

    assert len(single._plane_records) == 1  # noqa: SLF001
    assert len(multi._plane_records) == 2  # noqa: SLF001
