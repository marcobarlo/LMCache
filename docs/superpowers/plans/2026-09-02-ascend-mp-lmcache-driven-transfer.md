# Ascend MP lmcache_driven Transfer Path (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MP-mode `lmcache_driven` transfer path functional on Ascend NPU (explicit opt-in via `LMCACHE_MP_TRANSFER_MODE=lmcache_driven`), pure Python, no csrc changes.

**Architecture:** Mirror the MUSA platform (`lmcache/v1/platform/musa/`) — the existing precedent for a CuPy-less, no-native-host-callback platform on the LMCacheDriven path. All upstream changes are additive under `lmcache/v1/platform/npu/`; shared code (`GPUCacheContext`, the cache-context factory, `worker_transfer`, the event bus) is untouched. Stream-ordered completion is preserved by synchronizing the stream before the immediate-enqueue fallback (MUSA pattern).

**Tech Stack:** Python 3, torch 2.10 + torch_npu 2.10 (CANN 9.1 on the dev box), pytest, msgspec. The LMCache-Ascend plugin provides `AscendIPCWrapper` (already registered) and the pinned server pool.

**Spec:** `docs/superpowers/specs/2026-09-02-ascend-mp-lmcache-driven-transfer-design.md`

## Global Constraints

- Repos: upstream work in `/mnt/sdb/jjy/LMCache` (branch `support_ascend_mp_mode_dev_3`); plugin work in `/mnt/sdb/jjy/LMCache-Ascend` (same branch name). Never edit plugin-generated files (`_build_info.py`, `_version.py`).
- Every new Python file starts with `# SPDX-License-Identifier: Apache-2.0`.
- Import order per isort with section comments: `# Future`, `# Standard`, `# Third Party`, `# First Party` (upstream) / `# Local` (plugin). ruff line length 88.
- Type hints on all functions; no `Any` where avoidable; validation via `if/raise`, never `assert`.
- Docstring policy (user decision): CUDA/MUSA style — module and class one-liners, full `Args`/`Returns`/`Raises` on public methods, none on trivial private helpers.
- In the plugin repo, every line that diverges from the upstream original carries `# LMC-A: <reason>`.
- Upstream CPU-runnable NPU tests carry `pytestmark = pytest.mark.no_shared_allocator` (the autouse 5 GB pinned-allocator fixture fails on NPU hosts with the plugin installed — error 107002).
- Test invocation on the NPU dev box (upstream repo):
  `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest <file> -q -p noshared` after creating
  `/tmp/lmc_test_plugin/noshared.py` with `pytest_collection_modifyitems` marking every item `no_shared_allocator` (recipe in project memory).
- Plugin tests run with `LMCACHEPATH=/mnt/sdb/jjy/LMCache python3 -m pytest <file> -v` from `/mnt/sdb/jjy/LMCache-Ascend`.
- Commits: upstream uses `[MP][Ascend] <summary>` style; end every commit message with `Co-Authored-By: Claude <noreply@anthropic.com>`.
- Do NOT flip the AUTO transfer route (`worker_transfer.py` AUTO still sends `npu` to engine-driven — that is Phase 2).
- Reference templates to mirror: `lmcache/v1/platform/musa/cache_context.py`, `lmcache/v1/platform/musa/event_ipc.py`, `lmcache/v1/platform/musa/device_ops.py:559-616`, `lmcache/v1/platform/musa/__init__.py`.

---

### Task 1: NpuEventIPCBackend + DeviceSpec wiring

**Files:**
- Create: `lmcache/v1/platform/npu/event_ipc.py`
- Modify: `lmcache/v1/platform/npu/__init__.py`
- Test: `tests/v1/platform/npu/test_npu_event_ipc.py`

**Interfaces:**
- Consumes: `DefaultEventIPCBackend(event_module, device_type)` from `lmcache/v1/platform/base/event_ipc.py` (constructor validates nothing; `check_event_support(device)` raises when the module's `Event` lacks `interprocess=`/`from_ipc_handle`).
- Produces: `NpuEventIPCBackend(event_module: Any | None = None)` and module function `_torch_npu_module() -> Any`; `NpuDeviceSpec.event_ipc_backend` property returning a cached instance.

- [ ] **Step 1: Write the failing test**

Create `tests/v1/platform/npu/test_npu_event_ipc.py`:

```python
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


def test_device_spec_exposes_cached_event_backend() -> None:
    spec = NpuDeviceSpec()
    first = spec.event_ipc_backend
    assert first.device_type == "npu"
    assert spec.event_ipc_backend is first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest tests/v1/platform/npu/test_npu_event_ipc.py -q -p noshared`
Expected: FAIL — `ModuleNotFoundError: No module named 'lmcache.v1.platform.npu.event_ipc'`

- [ ] **Step 3: Write the implementation**

Create `lmcache/v1/platform/npu/event_ipc.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU device-event IPC backend."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend


def _torch_npu_module() -> Any:
    """Return ``torch.npu``, failing closed when torch_npu is absent."""
    # Third Party
    import torch

    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError(
            "torch_npu is not installed; NPU interprocess events are unavailable."
        )
    return npu


class NpuEventIPCBackend(DefaultEventIPCBackend):
    """Event IPC backend over torch_npu interprocess events.

    torch_npu implements the CUDA-style event ABI (``interprocess=True``,
    ``ipc_handle``, ``from_ipc_handle``), so the default adapter applies
    directly; ``check_event_support`` fails closed on builds lacking it.
    """

    def __init__(self, event_module: Any | None = None) -> None:
        """Create an NPU event IPC backend.

        Args:
            event_module: Optional torch.npu-like module. Defaults to the
                installed torch_npu; injectable for tests.
        """
        super().__init__(
            event_module=(
                event_module if event_module is not None else _torch_npu_module()
            ),
            device_type="npu",
        )
```

In `lmcache/v1/platform/npu/__init__.py`, extend the `TYPE_CHECKING` block and add two members to `NpuDeviceSpec` (mirroring `MusaDeviceSpec`):

```python
if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend
```

Add as the first line of the class body:

```python
    _event_backend_cache: "EventIPCBackend | None" = None
```

Add after the `pin_memory_backend` property:

```python
    @property
    def event_ipc_backend(self) -> "EventIPCBackend":
        """Return the cached torch_npu event IPC backend."""
        backend = self._event_backend_cache
        if backend is None:
            # First Party
            from lmcache.v1.platform.npu.event_ipc import NpuEventIPCBackend

            backend = NpuEventIPCBackend()
            self._event_backend_cache = backend
        return backend
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest tests/v1/platform/npu/test_npu_event_ipc.py -q -p noshared`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add lmcache/v1/platform/npu/event_ipc.py lmcache/v1/platform/npu/__init__.py tests/v1/platform/npu/test_npu_event_ipc.py
git commit -m "[MP][Ascend] Add torch_npu event IPC backend

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: `_NpuHostCallbackStream` + `_TempNpuBuffer`

**Files:**
- Create: `lmcache/v1/platform/npu/cache_context.py` (this task adds the module helpers; Task 3 appends the context class)
- Test: `tests/v1/platform/npu/test_npu_cache_context.py`

**Interfaces:**
- Consumes: `KVLayerGroupsManager(kv_caches, engine_kv_formats=..., engine_group_infos=..., lmcache_tokens_per_chunk=..., separate_object_groups=...)` from `lmcache/v1/kv_layer_groups.py`; `EngineKVFormat` from `lmcache.lmcache_native`.
- Produces: `_NpuHostCallbackStream(stream)` with `.ptr -> int`, `.synchronize()`, `.launch_host_func(callback, arg=None)`; `_TempNpuBuffer(kv_layer_groups_manager, lmcache_tokens_per_chunk, device, max_batch_size=4)` with `get_temp_kernel_group_buffer(batch_idx, kernel_group_idx) -> torch.Tensor`, `get_temp_object_group_buffer(batch_idx, object_group_idx) -> torch.Tensor`, `get_kernel_group_shape_dtype(num_tokens, kernel_group_idx) -> tuple[torch.Size, torch.dtype]`, `max_batch_size -> int`, `get_cache_size_per_token() -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/v1/platform/npu/test_npu_cache_context.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for NPU cache-context helpers (no Ascend hardware)."""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
import lmcache.lmcache_native as lmcache_native
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.platform.npu.cache_context import (
    _NpuHostCallbackStream,
    _TempNpuBuffer,
)

pytestmark = pytest.mark.no_shared_allocator


class _FakeStream:
    def __init__(self) -> None:
        self.npu_stream = 0x1234
        self.synchronized = 0
        self.launched: list[tuple[Any, Any]] = []

    def synchronize(self) -> None:
        self.synchronized += 1


def test_host_callback_stream_ptr_reads_npu_stream() -> None:
    adapter = _NpuHostCallbackStream(_FakeStream())
    assert adapter.ptr == 0x1234


def test_host_callback_stream_launch_runs_callback_inline_after_sync() -> None:
    stream = _FakeStream()
    adapter = _NpuHostCallbackStream(stream)
    seen: list[Any] = []
    adapter.launch_host_func(seen.append, 42)
    assert seen == [42]
    assert stream.synchronized == 1


def _mla_groups_manager() -> KVLayerGroupsManager:
    # Per-layer [NB, BS, W] planes, MLA-style single plane per layer.
    num_layers, num_blocks, block_size, width = 4, 8, 16, 576
    tensors = [
        torch.zeros(num_blocks, block_size, width) for _ in range(num_layers)
    ]
    fmt = lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
    return KVLayerGroupsManager(
        tensors,
        engine_kv_formats=[fmt] * num_layers,
        engine_group_infos=(),
        lmcache_tokens_per_chunk=256,
    )


def test_temp_buffer_mla_staging_is_rank3() -> None:
    manager = _mla_groups_manager()
    buffer = _TempNpuBuffer(
        kv_layer_groups_manager=manager,
        lmcache_tokens_per_chunk=256,
        device=torch.device("cpu"),
        max_batch_size=4,
    )
    shape, dtype = buffer.get_kernel_group_shape_dtype(256, 0)
    # MLA staging is [L, slots, W]; chunk of 256 tokens over bs=16 -> 16 slots.
    assert tuple(shape) == (4, 16, 576)
    assert dtype == torch.float32
    view = buffer.get_temp_kernel_group_buffer(0, 0)
    assert tuple(view.shape) == (4, 16, 576)
    flat = buffer.get_temp_object_group_buffer(0, 0)
    assert flat.dtype == torch.uint8
    assert flat.nbytes == 4 * 16 * 576 * 4


def test_temp_buffer_object_group_view_is_contiguous_union() -> None:
    manager = _mla_groups_manager()
    buffer = _TempNpuBuffer(
        kv_layer_groups_manager=manager,
        lmcache_tokens_per_chunk=256,
        device=torch.device("cpu"),
        max_batch_size=2,
    )
    kg0 = buffer.get_temp_kernel_group_buffer(1, 0)
    flat = buffer.get_temp_object_group_buffer(1, 0)
    assert kg0.data_ptr() == flat.data_ptr()
    assert buffer.max_batch_size == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest tests/v1/platform/npu/test_npu_cache_context.py -q -p noshared`
Expected: FAIL — `ModuleNotFoundError: No module named 'lmcache.v1.platform.npu.cache_context'`

- [ ] **Step 3: Write the implementation**

Create `lmcache/v1/platform/npu/cache_context.py` with the module helpers (Task 3 appends `NpuCacheContext` to this file):

```python
# SPDX-License-Identifier: Apache-2.0
"""NPU cache context for LMCache-driven multiprocess transfer."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.utils import get_group_data_ptrs
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

logger = init_logger(__name__)


class _NpuHostCallbackStream:
    """Adapter for stream-ordered callback call sites.

    CUDA contexts expose a CuPy stream as ``cupy_stream``. torch_npu has no
    CuPy equivalent, so this adapter provides ordered callback and
    synchronization operations over a torch_npu stream and exposes its
    pointer through the existing platform stream contract.
    """

    def __init__(self, stream: object) -> None:
        self._stream = stream

    def synchronize(self) -> None:
        """Wait for all work submitted to the wrapped NPU stream."""
        synchronize = getattr(self._stream, "synchronize", None)
        if not callable(synchronize):
            raise RuntimeError("NPU stream does not support synchronization")
        synchronize()

    @property
    def ptr(self) -> int:
        """Return the wrapped NPU stream pointer for platform call sites."""
        pointer = getattr(self._stream, "npu_stream", None)
        if pointer is None:
            raise RuntimeError("NPU stream does not expose a stream pointer")
        return int(pointer)

    def launch_host_func(self, callback: Any, arg: Any = None) -> None:
        """Schedule or run ``callback(arg)``.

        If the backend stream exposes ``launch_host_func`` directly, delegate
        to it. Otherwise synchronize the stream before running the callback on
        the current thread.
        """
        launch_host_func = getattr(self._stream, "launch_host_func", None)
        if callable(launch_host_func):
            launch_host_func(callback, arg)
            return
        self.synchronize()
        callback(arg)


class _TempNpuBuffer:
    """Owns NPU staging buffers for MP block-transfer batches."""

    def __init__(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int = 4,
    ) -> None:
        self._kv_groups_manager = kv_layer_groups_manager
        self._lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self._max_batch_size = max_batch_size

        self._temp_buffer = torch.empty(
            self._get_size_for_single_batch() * max_batch_size,
            dtype=torch.uint8,
            device=device,
        )
        self._offset_map_kernel_group_only: dict[tuple[int, int], tuple[int, int]] = {}
        self._offset_map_object_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        offset = 0
        for batch_idx in range(max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_start_offset = offset
                object_group_size = 0
                object_group = self._kv_groups_manager.object_groups[object_group_idx]
                for kernel_group_idx in object_group.kernel_group_indices:
                    size = self._get_size_for_kernel_group(kernel_group_idx)
                    self._offset_map_kernel_group_only[
                        (batch_idx, kernel_group_idx)
                    ] = (
                        offset,
                        size,
                    )
                    offset += size
                    object_group_size += size

                self._offset_map_object_group_only[(batch_idx, object_group_idx)] = (
                    object_group_start_offset,
                    object_group_size,
                )

        self._shape_cache_kernel_group: dict[int, tuple[torch.Size, torch.dtype]] = {}
        for kernel_group_idx in range(self._kv_groups_manager.num_kernel_groups):
            shape = self._get_shape_for_kernel_group(
                self._lmcache_tokens_per_chunk,
                kernel_group_idx,
            )
            group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
            self._shape_cache_kernel_group[kernel_group_idx] = (shape, group.dtype)

    @property
    def max_batch_size(self) -> int:
        """Return the number of chunks that fit in the staging buffer."""
        return self._max_batch_size

    def get_temp_kernel_group_buffer(
        self,
        batch_idx: int,
        kernel_group_idx: int,
    ) -> torch.Tensor:
        """Return a typed staging view for a batch/kernel-group pair."""
        key = (batch_idx, kernel_group_idx)
        if key not in self._offset_map_kernel_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or kernel_group_idx {kernel_group_idx}"
            )

        offset, size = self._offset_map_kernel_group_only[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._temp_buffer[offset : offset + size].view(dtype).view(shape)

    def get_temp_object_group_buffer(
        self,
        batch_idx: int,
        object_group_idx: int,
    ) -> torch.Tensor:
        """Return a flat ``uint8`` staging view for a batch/object-group pair."""
        key = (batch_idx, object_group_idx)
        if key not in self._offset_map_object_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or object_group_idx {object_group_idx}"
            )

        offset, size = self._offset_map_object_group_only[key]
        return self._temp_buffer[offset : offset + size]

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Return ``(shape, dtype)`` for a kernel group and token count."""
        _, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._get_shape_for_kernel_group(num_tokens, kernel_group_idx), dtype

    def get_cache_size_per_token(self) -> int:
        """Return total cache bytes per logical token across all groups."""
        return self._get_size_for_single_batch() // self._lmcache_tokens_per_chunk

    def _get_shape_for_kernel_group(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> torch.Size:
        if num_tokens % self._lmcache_tokens_per_chunk != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be a multiple of "
                f"lmcache_tokens_per_chunk ({self._lmcache_tokens_per_chunk})"
            )

        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        num_chunks = num_tokens // self._lmcache_tokens_per_chunk
        num_slots = (
            self._kv_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)
            * num_chunks
        )
        if group.engine_kv_format in (
            EngineKVFormat.NL_X_NB_BS_HS,
            EngineKVFormat.NL_X_TWO_X_NB_BS_HS,
        ):
            return torch.Size((group.num_layers, num_slots, group.hidden_dim_size))
        sd = group.shape_desc
        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def _get_size_for_kernel_group(self, kernel_group_idx: int) -> int:
        shape = self._get_shape_for_kernel_group(
            self._lmcache_tokens_per_chunk,
            kernel_group_idx,
        )
        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        return shape.numel() * group.dtype.itemsize

    def _get_size_for_object_group(self, object_group_idx: int) -> int:
        object_group = self._kv_groups_manager.object_groups[object_group_idx]
        return sum(
            self._get_size_for_kernel_group(kernel_group_idx)
            for kernel_group_idx in object_group.kernel_group_indices
        )

    def _get_size_for_single_batch(self) -> int:
        return sum(
            self._get_size_for_object_group(object_group_idx)
            for object_group_idx in range(self._kv_groups_manager.num_object_groups)
        )
```

Note: `NL_X_TWO_X_NB_BS_HS` is guarded into the 3D staging branch — the torch-fallback MLA transfer consumes the object tensor as `obj[:, offset:token_end]` (3D `[L, slots, W]`). If `EngineKVFormat` on the installed native build lacks the name, drop only that literal from the tuple (the detection tests skip the same way, `tests/v1/gpu_connector/test_kv_format_detection.py:179`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest tests/v1/platform/npu/test_npu_cache_context.py -q -p noshared`
Expected: PASS (4 tests). If `KVLayerGroupsManager` rejects positional tensors, read its `__init__` in `lmcache/v1/kv_layer_groups.py` and adapt the test helper call only — the production code does not change.

- [ ] **Step 5: Commit**

```bash
git add lmcache/v1/platform/npu/cache_context.py tests/v1/platform/npu/test_npu_cache_context.py
git commit -m "[MP][Ascend] Add NPU staging buffer and host-callback stream adapter

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: `NpuCacheContext` + `create_cache_context` wiring

**Files:**
- Modify: `lmcache/v1/platform/npu/cache_context.py` (append the context class)
- Modify: `lmcache/v1/platform/npu/__init__.py`
- Test: `tests/v1/platform/npu/test_npu_cache_context.py` (append)

**Interfaces:**
- Consumes: Task 2's `_NpuHostCallbackStream`, `_TempNpuBuffer`; `unwrap`-style `[w.to_tensor() for w in kv_caches]`; `normalize_and_discover_per_layer_formats(unwrapped, layer_indices, engine_type, layout_hints)` and `get_device(tensors)` from `lmcache/v1/gpu_connector/utils`; `engine_group_layer_indices` from `lmcache/v1/multiprocess/group_view`; `BaseCacheContext.__init__(kv_caches=..., device=..., num_layers=..., kv_layer_groups_manager=..., block_ids_buffer=..., lmcache_tokens_per_chunk=...)`.
- Produces: `NpuCacheContext(kv_caches: KVCache, lmcache_tokens_per_chunk: int = 256, layout_hints: LayoutHints | None = None, engine_group_infos: Sequence[EngineGroupInfo] = (), engine_type: EngineType = EngineType.VLLM, separate_object_groups: bool = False, full_sw_kv: bool = False)` with properties `stream`, `cupy_stream`, `get_kernel_group_kv_pointers(idx)`, `get_temp_kernel_group_buffer(batch_idx, kg_idx)`, `get_temp_object_group_buffer(batch_idx, og_idx)`, `max_batch_size`, `get_kernel_group_shape_dtype(num_tokens, kg_idx)`, `cache_size_per_token()`, `close()`; `NpuDeviceSpec.create_cache_context(*args, **kwargs)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/v1/platform/npu/test_npu_cache_context.py`:

```python
def test_close_synchronizes_before_releasing_ipc_owners() -> None:
    """Context close waits for transfers, releases owners, and is idempotent."""
    # First Party
    from lmcache.v1.platform.npu.cache_context import NpuCacheContext

    calls: list[str] = []

    class _Stream:
        def synchronize(self) -> None:
            calls.append("synchronize")

    class _Wrapper:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append(self.name)

    class _TestContext(NpuCacheContext):
        def __init__(self) -> None:
            self.stream_ = _Stream()  # type: ignore[assignment]
            self._ipc_wrappers = (  # type: ignore[assignment]
                _Wrapper("first"),
                _Wrapper("second"),
            )

    context = _TestContext()

    context.close()
    context.close()

    assert calls == ["synchronize", "first", "second"]


def test_device_spec_creates_npu_cache_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spec hook constructs NpuCacheContext with forwarded arguments."""
    # First Party
    from lmcache.v1.platform.npu import NpuDeviceSpec
    from lmcache.v1.platform.npu import cache_context as npu_cache_context

    class _Sentinel:
        pass

    created: dict[str, object] = {}

    def _fake_init(
        self: object,
        kv_caches: object,
        lmcache_tokens_per_chunk: int = 256,
        **kwargs: object,
    ) -> None:
        created["kv_caches"] = kv_caches
        created["chunk"] = lmcache_tokens_per_chunk

    monkeypatch.setattr(npu_cache_context.NpuCacheContext, "__init__", _fake_init)
    spec = NpuDeviceSpec()
    context = spec.create_cache_context([_Sentinel()], 128)
    assert isinstance(context, npu_cache_context.NpuCacheContext)
    assert isinstance(created["kv_caches"], list)
    assert created["chunk"] == 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest tests/v1/platform/npu/test_npu_cache_context.py -q -p noshared`
Expected: the two new tests FAIL — `ImportError: cannot import name 'NpuCacheContext'`

- [ ] **Step 3: Write the implementation**

Append to `lmcache/v1/platform/npu/cache_context.py`:

```python
class NpuCacheContext(BaseCacheContext):
    """Cache context for NPU-backed KV tensors in MP handle mode."""

    device_type = "npu"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = False,
        full_sw_kv: bool = False,
    ) -> None:
        """Build an NPU cache context from IPC-wrapped KV tensors.

        Args:
            kv_caches: NPU IPC wrappers received during server registration.
            lmcache_tokens_per_chunk: Tokens per LMCache object.
            layout_hints: Optional KV layout hints from the serving engine.
            engine_group_infos: Engine-neutral group metadata.
            engine_type: Serving engine that produced the KV cache.
            separate_object_groups: Whether to split object groups by window.
            full_sw_kv: Whether sliding-window groups transfer their full KV.

        Raises:
            ValueError: If reconstructed tensors are not NPU tensors.
        """
        self._ipc_wrappers: tuple[Any, ...] = tuple(kv_caches)
        try:
            self._initialize(
                kv_caches,
                lmcache_tokens_per_chunk,
                layout_hints,
                engine_group_infos,
                engine_type,
                separate_object_groups,
                full_sw_kv,
            )
        except BaseException:
            self.close()
            raise

    def _initialize(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int,
        layout_hints: LayoutHints | None,
        engine_group_infos: Sequence[EngineGroupInfo],
        engine_type: EngineType,
        separate_object_groups: bool,
        full_sw_kv: bool,
    ) -> None:
        """Initialize reconstructed tensors and NPU transfer resources."""
        unwrapped = [wrapper.to_tensor() for wrapper in kv_caches]
        discovered, engine_kv_formats = normalize_and_discover_per_layer_formats(
            unwrapped,
            engine_group_layer_indices(engine_group_infos),
            engine_type,
            layout_hints,
        )
        if not isinstance(discovered, list):
            raise ValueError("NpuCacheContext requires a list-based KV layout")
        self.device_ = get_device(discovered)
        if self.device_.type != "npu":
            raise ValueError(
                f"NpuCacheContext expected NPU tensors, got {self.device_.type!r}"
            )

        kv_layer_groups_manager = KVLayerGroupsManager(
            discovered,
            engine_kv_formats=engine_kv_formats,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            separate_object_groups=separate_object_groups,
        )
        if full_sw_kv:
            kv_layer_groups_manager.enable_full_sw_kv()

        block_ids_buffer = torch.empty(
            1 << 20,
            dtype=torch.long,
            device=self.device_,
        )

        super().__init__(
            kv_caches=discovered,
            device=self.device_,
            num_layers=len(engine_kv_formats),
            kv_layer_groups_manager=kv_layer_groups_manager,
            block_ids_buffer=block_ids_buffer,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        self.group_kv_pointers_: list[torch.Tensor] = []
        for group_idx, group in enumerate(self.kv_layer_groups_manager_.kernel_groups):
            pointers = get_group_data_ptrs(
                self.kv_caches_,
                self.get_engine_kv_format(group_idx),
                group.layer_indices,
            )
            self.group_kv_pointers_.append(
                torch.tensor(pointers, dtype=torch.int64, device=self.device_)
            )

        self._temp_buffer = _TempNpuBuffer(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=self.device_,
            max_batch_size=4,
        )
        self.stream_ = torch_dev.Stream(device=self.device_)
        self.host_callback_stream_ = _NpuHostCallbackStream(self.stream_)

        logger.debug(
            "NpuCacheContext: %d layers, %d blocks, dtype=%s",
            self.num_layers_,
            self.num_blocks,
            self.kv_layer_groups_manager_.kernel_groups[0].dtype,
        )

    def close(self) -> None:
        """Synchronize transfers and release receiver-side IPC owners."""
        wrappers = self._ipc_wrappers
        if not wrappers:
            return

        stream = getattr(self, "stream_", None)
        synchronize = getattr(stream, "synchronize", None)
        if callable(synchronize):
            synchronize()

        kv_tensors = getattr(self, "kv_caches_", None)
        if isinstance(kv_tensors, list):
            kv_tensors.clear()

        for wrapper in wrappers:
            close = getattr(wrapper, "close", None)
            if callable(close):
                close()
        self._ipc_wrappers = ()

    @property
    def stream(self) -> Any:
        """Return the NPU stream used for transfer work."""
        return self.stream_

    @property
    def cupy_stream(self) -> _NpuHostCallbackStream:
        """Return a host-callback stream adapter for shared MP code paths."""
        return self.host_callback_stream_

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Return packed, process-local NPU pointers for a kernel group.

        Args:
            kernel_group_idx: Index of the requested kernel group.

        Returns:
            A one-dimensional ``int64`` tensor in kernel order.
        """
        return self.group_kv_pointers_[kernel_group_idx]

    def get_temp_kernel_group_buffer(
        self,
        batch_idx: int,
        kernel_group_idx: int,
    ) -> torch.Tensor:
        """Return the NPU staging buffer for a batch/kernel-group pair.

        Args:
            batch_idx: Index within the current transfer batch.
            kernel_group_idx: Index of the kernel group to transfer.

        Returns:
            A typed view into the NPU staging allocation.
        """
        return self._temp_buffer.get_temp_kernel_group_buffer(
            batch_idx,
            kernel_group_idx,
        )

    @property
    def max_batch_size(self) -> int:
        """Return the maximum number of chunks transferred per batch."""
        return self._temp_buffer.max_batch_size

    def get_temp_object_group_buffer(
        self,
        batch_idx: int,
        object_group_idx: int,
    ) -> torch.Tensor:
        """Return the NPU staging buffer for a batch/object-group pair.

        Args:
            batch_idx: Index within the current transfer batch.
            object_group_idx: Index of the object group to transfer.

        Returns:
            A flat view covering the object's NPU staging allocation.
        """
        return self._temp_buffer.get_temp_object_group_buffer(
            batch_idx,
            object_group_idx,
        )

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Return the shape and dtype for a kernel-group allocation.

        Args:
            num_tokens: Number of tokens represented by the allocation.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The allocation shape and element dtype.
        """
        return self._temp_buffer.get_kernel_group_shape_dtype(
            num_tokens,
            kernel_group_idx,
        )

    def cache_size_per_token(self) -> int:
        """Return total cache bytes per logical token across all groups."""
        return self._temp_buffer.get_cache_size_per_token()
```

Extend the imports at the top of `cache_context.py`:

```python
# Standard
from collections.abc import Sequence
from typing import Any

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_device,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.group_view import engine_group_layer_indices
from lmcache.v1.platform.base.cache_context import BaseCacheContext

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
```

(keep `get_group_data_ptrs` and the existing imports; drop nothing).

In `lmcache/v1/platform/npu/__init__.py`, add to `NpuDeviceSpec` after `event_ipc_backend`:

```python
    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        """Create the NPU cache context for LMCache-driven transfer."""
        # First Party
        from lmcache.v1.platform.npu.cache_context import NpuCacheContext

        return NpuCacheContext(*args, **kwargs)
```

and extend the module's `TYPE_CHECKING` import list and top imports with `Any` (`from typing import Any, TYPE_CHECKING`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest tests/v1/platform/npu/test_npu_cache_context.py tests/v1/platform/test_cache_context_dispatch.py -q -p noshared`
Expected: PASS (6 + dispatch suite unchanged)

- [ ] **Step 5: Commit**

```bash
git add lmcache/v1/platform/npu/cache_context.py lmcache/v1/platform/npu/__init__.py tests/v1/platform/npu/test_npu_cache_context.py
git commit -m "[MP][Ascend] Add NpuCacheContext for LMCache-driven transfer

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Stream-ordered completion recording in `NpuDeviceOps`

**Files:**
- Modify: `lmcache/v1/platform/npu/device_ops.py`
- Test: `tests/v1/platform/npu/test_npu_device_ops_ordering.py`

**Interfaces:**
- Consumes: `DeviceOps.record_completion_on_stream(stream_ptr, kind, payload)` / `record_event_on_stream(...)` torch baseline (immediate enqueue into module-global buffers drained by `drain_recorded_completions` / `drain_recorded_events`).
- Produces: `_synchronize_npu_stream_pointer(stream_ptr: int) -> None` (module-level in `lmcache/v1/platform/npu/device_ops.py`); `NpuDeviceOps.record_completion_on_stream` / `NpuDeviceOps.record_event_on_stream` overrides that sync first.

- [ ] **Step 1: Write the failing test**

Create `tests/v1/platform/npu/test_npu_device_ops_ordering.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest tests/v1/platform/npu/test_npu_device_ops_ordering.py -q -p noshared`
Expected: FAIL — `AttributeError: ... has no attribute '_synchronize_npu_stream_pointer'` (or type-error on the override signature)

- [ ] **Step 3: Write the implementation**

Extend `lmcache/v1/platform/npu/device_ops.py`. New module-level helper above the class:

```python
def _synchronize_npu_stream_pointer(stream_ptr: int) -> None:
    """Synchronize a raw NPU stream pointer through the CANN runtime.

    Args:
        stream_ptr: Raw ``aclrtStream`` handle.

    Raises:
        TypeError: If ``stream_ptr`` is not an int.
        RuntimeError: If the acl runtime module is unavailable or reports
            an error.
    """
    if not isinstance(stream_ptr, int):
        raise TypeError("NPU stream pointer must be an int")
    try:
        # Third Party
        from acl import rt as aclrt
    except ImportError as exc:
        raise RuntimeError(
            f"Unable to synchronize NPU stream pointer {stream_ptr}: "
            "the acl runtime module is unavailable"
        ) from exc
    try:
        ret = aclrt.synchronize_stream(stream_ptr)
    except Exception as exc:
        raise RuntimeError(
            f"aclrtSynchronizeStream raised for stream {stream_ptr}"
        ) from exc
    if isinstance(ret, int) and ret != 0:
        raise RuntimeError(
            f"aclrtSynchronizeStream failed with error {ret} for stream "
            f"{stream_ptr}"
        )
```

Add two overrides to `NpuDeviceOps` (after `ensure_native`):

```python
    def record_completion_on_stream(
        self,
        stream_ptr: int,
        kind: str,
        payload: bytes,
    ) -> None:
        """Publish a completion only after prior NPU stream work finishes.

        torch_npu does not expose the CUDA host-callback primitive used by
        LMCache's native completion recorder. Synchronizing here preserves
        the storage ownership contract until the plugin ships an async
        callback backend (its native binding shadows this override via
        ``bind_native``).

        Args:
            stream_ptr: Raw NPU stream pointer from the generic recorder path.
            kind: Completion handler key.
            payload: Encoded completion payload.

        Raises:
            RuntimeError: If the NPU stream cannot be synchronized.
        """
        _synchronize_npu_stream_pointer(stream_ptr)
        super().record_completion_on_stream(0, kind, payload)

    def record_event_on_stream(
        self,
        stream_ptr: int,
        event_type_name: str,
        session_id: str,
        str_metadata: dict[str, str],
        int_metadata: dict[str, int],
    ) -> None:
        """Record an event only after prior NPU stream work finishes.

        Args:
            stream_ptr: Raw NPU stream pointer from the generic recorder path.
            event_type_name: Serialized event type.
            session_id: Session associated with the event.
            str_metadata: String-valued event metadata.
            int_metadata: Integer-valued event metadata.

        Raises:
            RuntimeError: If the NPU stream cannot be synchronized.
        """
        _synchronize_npu_stream_pointer(stream_ptr)
        super().record_event_on_stream(
            0,
            event_type_name,
            session_id,
            str_metadata,
            int_metadata,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest tests/v1/platform/npu/test_npu_device_ops_ordering.py -q -p noshared`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add lmcache/v1/platform/npu/device_ops.py tests/v1/platform/npu/test_npu_device_ops_ordering.py
git commit -m "[MP][Ascend] Keep completion recording stream-ordered on NPU

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: NPU-gated device tests (real hardware)

**Files:**
- Test: `tests/v1/platform/npu/test_npu_event_ipc_device.py`

**Interfaces:**
- Consumes: Tasks 1-4 (`NpuEventIPCBackend`, `_TempNpuBuffer`); real `torch.npu`.
- Produces: hardware proof for the spec's open items (cross-process events; real-device staging).

- [ ] **Step 1: Write the test**

Create `tests/v1/platform/npu/test_npu_event_ipc_device.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""NPU-gated tests: cross-process interprocess events and device staging."""

# Standard
import multiprocessing as mp

# Third Party
import pytest
import torch

# First Party
import lmcache.lmcache_native as lmcache_native
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.platform.npu.cache_context import _TempNpuBuffer

pytestmark = [
    pytest.mark.npu,
    pytest.mark.no_shared_allocator,
]

pytestmark_skip = not (
    hasattr(torch, "npu") and torch.npu.is_available()
)

requires_npu = pytest.mark.skipif(
    pytestmark_skip, reason="Ascend NPU hardware is required"
)


def _child_event_producer(conn) -> None:
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        payload = torch.full((1024,), 7.0, device="npu")
        payload.mul_(2)
    event = torch.npu.Event(enable_timing=False, interprocess=True)
    event.record(stream)
    conn.send({"handle": event.ipc_handle(), "expected": 14.0})
    conn.recv()  # keep the producer alive until the parent is done


@requires_npu
def test_cross_process_event_import_and_sync() -> None:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=_child_event_producer, args=(child_conn,))
    process.start()
    try:
        message = parent_conn.recv()
        imported = torch.npu.Event.from_ipc_handle(
            torch.device("npu:0"), message["handle"]
        )
        imported.synchronize()
        assert imported.query() is True
    finally:
        parent_conn.send("done")
        process.join(timeout=30)
    assert process.exitcode == 0


@requires_npu
def test_backend_roundtrip_on_device() -> None:
    from lmcache.v1.platform.npu import NpuDeviceSpec

    backend = NpuDeviceSpec().event_ipc_backend
    device = torch.device("npu:0")
    backend.check_event_support(device)
    event = backend.create_event(device)
    stream = torch.npu.Stream()
    payload = torch.ones(8, device=device)
    payload.add_(1)
    backend.record_event(event, stream)
    backend.synchronize_event(event, device)
    assert backend.query_event(event) is True
    handle = backend.export_event(event, device)
    imported = backend.import_event(handle, device)
    assert imported is not event


@requires_npu
def test_temp_buffer_allocates_on_device() -> None:
    device = torch.device("npu:0")
    num_layers, num_blocks, block_size, width = 4, 8, 16, 576
    tensors = [
        torch.zeros(num_blocks, block_size, width, device=device)
        for _ in range(num_layers)
    ]
    manager = KVLayerGroupsManager(
        tensors,
        engine_kv_formats=(
            [lmcache_native.EngineKVFormat.NL_X_NB_BS_HS] * num_layers
        ),
        engine_group_infos=(),
        lmcache_tokens_per_chunk=256,
    )
    buffer = _TempNpuBuffer(
        kv_layer_groups_manager=manager,
        lmcache_tokens_per_chunk=256,
        device=device,
        max_batch_size=4,
    )
    view = buffer.get_temp_kernel_group_buffer(0, 0)
    assert view.device.type == "npu"
    view.fill_(1.0)
    flat = buffer.get_temp_object_group_buffer(0, 0)
    assert flat.view(torch.float32).sum().item() == float(view.numel())
```

- [ ] **Step 2: Run on the dev box**

Run: `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest tests/v1/platform/npu/test_npu_event_ipc_device.py -v -p noshared`
Expected: 3 PASS on the 910B box (skipped elsewhere). If `from_ipc_handle` in the same process fails with driver error 17, only `test_backend_roundtrip_on_device`'s import path is affected — restrict that assertion to cross-process (already covered by the first test) and keep create/export/synchronize.

- [ ] **Step 3: Commit**

```bash
git add tests/v1/platform/npu/test_npu_event_ipc_device.py
git commit -m "test(mp/ascend): NPU-gated event IPC and staging device tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Design doc under `docs/design/v1/platform/npu/`

**Files:**
- Create: `docs/design/v1/platform/npu/mp_lmcache_driven_transfer.md`

**Interfaces:**
- Consumes: the spec and Tasks 1-4.
- Produces: repo-convention design documentation (CLAUDE.md requires `docs/design/<path>/` mirroring `lmcache/<path>/`).

- [ ] **Step 1: Write the doc**

```markdown
# NPU Platform: LMCache-Driven MP Transfer

`lmcache/v1/platform/npu/` implements the Ascend NPU side of the MP-mode
`lmcache_driven` handle-transfer path. It mirrors the MUSA platform (the
precedent for a CuPy-less platform on this path); all kernels stay in the
external `lmcache_ascend` plugin and are layered on by `NpuDeviceOps` via
`DeviceOps.bind_native`.

## Components

| Module | Responsibility |
|---|---|
| `event_ipc.py` | `NpuEventIPCBackend` — torch_npu implements the CUDA-style interprocess event ABI (`interprocess=True`, `ipc_handle`, `from_ipc_handle`), so the shared `DefaultEventIPCBackend` adapter applies directly. `check_event_support` fails closed on builds lacking the ABI. |
| `cache_context.py` | `NpuCacheContext` (subclass of `BaseCacheContext`) — imports worker KV mappings from `AscendIPCWrapper`, builds per-kernel-group pointer tables, owns `_TempNpuBuffer` staging and the transfer stream. `_NpuHostCallbackStream` adapts the torch_npu stream to the `cupy_stream` contract (`.ptr` from `npu_stream`; `launch_host_func` degrades to synchronize-then-run). |
| `device_ops.py` | `NpuDeviceOps` binds `lmcache_ascend.c_ops` and keeps completion/event recording stream-ordered: `_synchronize_npu_stream_pointer` (via `acl.rt.synchronize_stream`) runs before the immediate-enqueue fallback, preserving the `finish_write` storage-ownership contract until the plugin ships a native `aclrtLaunchCallback` recorder. |

## Staging layout

`_TempNpuBuffer` allocates one flat `uint8` buffer per
`max_batch_size` chunks with two offset maps — `(batch, kernel_group)` and
`(batch, object_group)`. Per-layer MLA-family formats
(`NL_X_NB_BS_HS`, `NL_X_TWO_X_NB_BS_HS`) stage as rank-3
`[L, slots, W]`; other formats use the rank-4
`(kv_size, L, slots, W)` layout. Kernel-group buffers are contiguous inside
their object group (the staging memcpy contract).

## Enabling

The path is opt-in: set `LMCACHE_MP_TRANSFER_MODE=lmcache_driven`.
AUTO still routes `npu` to engine-driven. The worker-side wrapper
(`AscendIPCWrapper`) is registered by the LMCache-Ascend plugin on
`NpuDeviceSpec.ipc_wrapper_cls`.

Spec: `docs/superpowers/specs/2026-09-02-ascend-mp-lmcache-driven-transfer-design.md`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/design/v1/platform/npu/mp_lmcache_driven_transfer.md
git commit -m "docs(design): NPU platform LMCache-driven MP transfer

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Plugin — collect-ignore carve-out and `test_custom_types` NPU fix

Work in `/mnt/sdb/jjy/LMCache-Ascend`.

**Files:**
- Modify: `tests/conftest.py:11-13`
- Modify: `tests/v1/multiprocess/test_custom_types.py:66-69,87`

**Interfaces:**
- Consumes: upstream Tasks 1-4 (must be present in the checkout at `LMCACHEPATH`).
- Produces: `tests/v1/multiprocess/test_custom_types.py` and (Task 8) `test_npu_lmcache_driven_e2e.py` collected and passing on NPU.

- [ ] **Step 1: Update the collect-ignore block**

Replace in `tests/conftest.py`:

```python
# Skip multiprocess tests entirely — NPU does not support IPC sharing
collect_ignore_glob = ["v1/multiprocess/test_*.py"]
```

with:

```python
# LMC-A: multiprocess tests were skipped wholesale ("NPU does not support IPC
# sharing"). Cross-process IPC now works (AscendIPCWrapper + torch_npu event
# handles), so keep the two NPU-capable files collected and skip the rest,
# which still exercise CUDA-only paths.
import glob as _glob
import os as _os

_MP_KEEP = {
    "test_custom_types.py",
    "test_npu_lmcache_driven_e2e.py",
}
_MP_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "v1", "multiprocess")
)
collect_ignore = sorted(
    path
    for path in _glob.glob(_os.path.join(_MP_DIR, "test_*.py"))
    if _os.path.basename(path) not in _MP_KEEP
)
```

- [ ] **Step 2: Fix the device selection in `test_custom_types.py`**

Change the skip guard (lines 66-69):

```python
@pytest.mark.skipif(
    not torch.cuda.is_available(),  # LMC-A: run on npu, not cuda
    reason="NPU is required for IPCWrapper multiprocessing tests",
)
```

to:

```python
@pytest.mark.skipif(
    not torch.npu.is_available(),  # LMC-A: run on npu, not cuda
    reason="NPU is required for IPCWrapper multiprocessing tests",
)
```

and the tensor creation (line 87):

```python
        tensor = torch.full(
            (2, 3), fill_value=float(i + 1), dtype=torch.float32, device="cuda"
        )
```

to:

```python
        tensor = torch.full(
            (2, 3),
            fill_value=float(i + 1),
            dtype=torch.float32,
            device="npu",  # LMC-A: run on npu, not cuda
        )
```

- [ ] **Step 3: Run on the dev box**

Run: `cd /mnt/sdb/jjy/LMCache-Ascend && LMCACHEPATH=/mnt/sdb/jjy/LMCache python3 -m pytest tests/v1/multiprocess/test_custom_types.py -v`
Expected: all tests PASS or SKIP; `test_cudaipc_wrapper_multiprocess_serialization` PASSES (cross-process wrapper roundtrip incl. the worker-side `add_(1)` mutation visible in the parent). If the worker times out at 30 s, raise the join timeout to 60 s — torch_npu cold init on a loaded box is slow.

- [ ] **Step 4: Confirm the rest of the multiprocess suite stays ignored**

Run: `cd /mnt/sdb/jjy/LMCache-Ascend && LMCACHEPATH=/mnt/sdb/jjy/LMCache python3 -m pytest tests/v1/multiprocess/ --collect-only -q 2>/dev/null | tail -3`
Expected: only `test_custom_types.py` items collected (Task 8's file once it exists).

- [ ] **Step 5: Commit (in the plugin repo)**

```bash
git add tests/conftest.py tests/v1/multiprocess/test_custom_types.py
git commit -m "[MP] Collect the NPU-capable multiprocess tests again

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Plugin E2E — LMCacheDriven store/retrieve on 910B

Work in `/mnt/sdb/jjy/LMCache-Ascend`.

**Files:**
- Create: `tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py`

**Interfaces:**
- Consumes: upstream `LMCacheDrivenTransferModule` (driven with fakes exactly like `tests/v1/multiprocess/test_event_ipc_handle_path.py:191-298` in the upstream repo); `NpuCacheContext`, `NpuEventIPCBackend` (upstream Tasks 1-4); `AscendIPCWrapper` and `get_customized_encoder/decoder` (plugin `tests/v1/multiprocess/test_custom_types.py`).
- Produces: byte-level store/retrieve roundtrip across two real processes, MLA-tuple layout (`NL_X_TWO_X_NB_BS_HS`), worker on `npu:0` (and a second run on `npu:1` when the box exposes more than one device).

- [ ] **Step 1: Write the test**

Create `tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""E2E: LMCacheDriven store/retrieve across processes on Ascend NPU.

The worker process owns paged MLA-tuple KV planes, exports them through
AscendIPCWrapper plus an interprocess event; the parent drives the real
server-side LMCacheDrivenTransferModule (fake storage bus, real cache
context, real transfers, real events).
"""

# Standard
import multiprocessing as mp
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch
import torch_npu  # noqa: F401

# First Party
from tests.bootstrap import prepare_environment

prepare_environment()

# Third Party
import lmcache_ascend  # noqa: F401, E402  (registers AscendIPCWrapper)

# First Party
from lmcache.utils import EngineType  # noqa: E402
from lmcache.v1.gpu_connector.utils import LayoutHints  # noqa: E402
from lmcache.v1.multiprocess.custom_types import KVCache  # noqa: E402
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer  # noqa: E402
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (  # noqa: E402
    LMCacheDrivenTransferContext,
    create_transfer_context,
)
from lmcache_ascend.v1.multiprocess.custom_types import AscendIPCWrapper  # noqa: E402
from tests.v1.multiprocess.test_custom_types import (  # noqa: E402
    get_customized_decoder,
    get_customized_encoder,
)

NL = 4
NB = 16
BS = 16
CHUNK = 256  # BS * NB
W_LATENT = 128
W_ROPE = 16
HIDDEN = W_LATENT + W_ROPE

requires_npu = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU hardware is required"
)


def _worker(device_index: int, conn) -> None:
    torch.npu.set_device(device_index)
    device = f"npu:{device_index}"
    wrappers: list[AscendIPCWrapper] = []
    planes: list[torch.Tensor] = []
    for layer in range(NL):
        latent = torch.zeros(NB, BS, 1, W_LATENT, device=device)
        rope = torch.zeros(NB, BS, 1, W_ROPE, device=device)
        for block in range(NB):
            latent[block].fill_(float(layer * 1000 + block) % 251.0)
            rope[block].fill_(float((layer * 7 + block) % 13.0))
        planes.extend([latent, rope])
        wrappers.extend([AscendIPCWrapper(latent), AscendIPCWrapper(rope)])
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for plane in planes:
            plane.mul_(1.0)  # materialize writes on the worker stream
    event = torch.npu.Event(enable_timing=False, interprocess=True)
    event.record(stream)
    encoder = get_customized_encoder(type=list[AscendIPCWrapper])
    conn.send(
        {
            "wrappers": encoder.encode(wrappers),
            "producer": event.ipc_handle(),
            "device_index": device_index,
        }
    )
    conn.recv()  # hold the mappings until the parent finishes


class _NoopDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def register(self, *args: Any, **kwargs: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self, timeout: float = 0.0) -> None:
        pass


class _FakeMemoryObj:
    """MemoryObj stand-in hitting the plain-tensor memcpy branch."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.raw_tensor = tensor

    def get_size(self) -> int:
        return self.raw_tensor.numel() * self.raw_tensor.element_size()

    def parent(self) -> None:
        return None

    @property
    def data_ptr(self) -> int:
        return self.raw_tensor.data_ptr()


class _FakeStorageManager:
    def __init__(self) -> None:
        self.objects: dict[Any, _FakeMemoryObj] = {}

    def _bytes(self, layout_desc: Any) -> int:
        import math

        return sum(
            math.prod(shape) * dtype.itemsize
            for shape, dtype in zip(
                layout_desc.shapes, layout_desc.dtypes, strict=True
            )
        )

    def reserve_write(
        self, keys: list[Any], layout_desc: Any, mode: str
    ) -> dict[Any, _FakeMemoryObj]:
        reserved = {}
        for key in keys:
            tensor = torch.empty(
                self._bytes(layout_desc), dtype=torch.uint8, device="cpu"
            )
            obj = _FakeMemoryObj(tensor)
            reserved[key] = obj
            self.objects[key] = obj
        return reserved

    def finish_write(self, keys: list[Any], read_locks: int = 0) -> None:
        pass

    def finish_read_prefetched(self, keys: list[Any], read_locks: int = 0) -> None:
        pass

    def read_prefetched_results(self, keys: list[Any]) -> "_FakeReadWindow":
        return _FakeReadWindow([self.objects[k] for k in keys])


class _FakeReadWindow:
    def __init__(self, objects: list[_FakeMemoryObj]) -> None:
        self._objects = objects

    def __enter__(self) -> list[_FakeMemoryObj]:
        return self._objects

    def __exit__(self, *args: object) -> None:
        pass


def _expected_staging() -> torch.Tensor:
    """Rank-3 [L, NB*BS, W] expectation: latent then rope plane per layer."""
    expected = torch.empty(NL, NB * BS, HIDDEN, dtype=torch.float32)
    for layer in range(NL):
        for block in range(NB):
            expected[layer, block * BS : (block + 1) * BS, :W_LATENT] = (
                float(layer * 1000 + block) % 251.0
            )
            expected[layer, block * BS : (block + 1) * BS, W_LATENT:] = float(
                (layer * 7 + block) % 13.0
            )
    return expected


@requires_npu
def test_forced_mode_selects_lmcache_driven_context() -> None:
    """Explicit mode routes NPU tensors to the handle-transfer context."""
    tensors = {
        f"layer-{i}": torch.zeros(NB, BS, 1, W_LATENT, device="npu:0")
        for i in range(NL)
    }
    tensors.update(
        {
            f"layer-rope-{i}": torch.zeros(NB, BS, 1, W_ROPE, device="npu:0")
            for i in range(NL)
        }
    )
    context = create_transfer_context(tensors, mode="lmcache_driven")
    assert isinstance(context, LMCacheDrivenTransferContext)
    context.close()


@requires_npu
@pytest.mark.parametrize("device_index", [0, 1])
def test_lmcache_driven_store_and_retrieve_roundtrip(
    device_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if device_index >= torch.npu.device_count():
        pytest.skip("single-device box")

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=_worker, args=(device_index, child_conn))
    process.start()
    try:
        message = parent_conn.recv()
        decoder = get_customized_decoder(type=list[AscendIPCWrapper])
        kv_caches: KVCache = list(decoder.decode(message["wrappers"]))

        from lmcache.v1.platform.npu import NpuDeviceSpec
        from lmcache.v1.platform.npu.event_ipc import NpuEventIPCBackend

        cache_context = NpuDeviceSpec().create_cache_context(
            kv_caches,
            CHUNK,
            layout_hints=LayoutHints(kv_layout="NHD"),
            engine_group_infos=(),
            engine_type=EngineType.VLLM,
        )
        event_backend = NpuEventIPCBackend()
        storage_manager = _FakeStorageManager()

        module = lmcache_driven_transfer.LMCacheDrivenTransferModule(
            SimpleNamespace(
                chunk_size=CHUNK,
                storage_manager=storage_manager,
                event_bus=SimpleNamespace(
                    publish=lambda event: None,
                    publish_on_stream=lambda stream, event: None,
                    has_subscribers=lambda event_type: False,
                ),
                resolve_obj_keys=lambda key, group_ids: [[("chunk", 0)]],
            )
        )
        entry = lmcache_driven_transfer.ContextEntry(
            cache_context=cache_context,
            model_name="mla-e2e",
            world_size=1,
            event_backend=event_backend,
        )
        monkeypatch.setattr(
            module, "get_and_touch_context_entry", lambda iid: entry
        )
        key = SimpleNamespace(
            request_id="e2e",
            cache_salt="",
            worker_id=0,
            token_ids=list(range(CHUNK)),
            start=0,
            end=CHUNK,
        )

        block_ids = [list(range(NB))]
        monkeypatch.setattr(
            lmcache_driven_transfer, "DeviceHostFuncDispatcher", _NoopDispatcher
        )
        store_handle, stored = module.store(key, 1, block_ids, message["producer"])
        assert stored is True
        completion = event_backend.import_event(
            store_handle, torch.device(f"npu:{device_index}")
        )
        completion.synchronize()

        # Stored bytes equal the worker's plane contents in [L, tokens, W]
        # staging order (latent plane then rope plane per layer).
        stored_obj = storage_manager.objects[("chunk", 0)]
        host = stored_obj.raw_tensor.view(torch.float32)
        assert torch.allclose(host.view(NL, NB * BS, HIDDEN), _expected_staging())

        # Retrieve: mutate host bytes, scatter back into the same blocks.
        stored_obj.raw_tensor.view(torch.float32).add_(1.0)
        producer = event_backend.create_event(torch.device(f"npu:{device_index}"))
        event_backend.record_event(producer, torch.npu.current_stream())
        producer_handle = event_backend.export_event(
            producer, torch.device(f"npu:{device_index}")
        )
        retrieve_handle, retrieved = module.retrieve(
            key, 1, block_ids, producer_handle
        )
        assert retrieved is True
        done = event_backend.import_event(
            retrieve_handle, torch.device(f"npu:{device_index}")
        )
        done.synchronize()

        # Every imported plane element now carries the +1 mutation.
        expected = _expected_staging()
        for layer in range(NL):
            latent = kv_caches[2 * layer].to_tensor()
            rope = kv_caches[2 * layer + 1].to_tensor()
            for block in range(NB):
                exp_latent = expected[
                    layer, block * BS : (block + 1) * BS, :W_LATENT
                ] + 1.0
                exp_rope = expected[layer, block * BS : (block + 1) * BS, W_LATENT:] + 1.0
                assert torch.allclose(latent[block, :, 0, :], exp_latent)
                assert torch.allclose(rope[block, :, 0, :], exp_rope)
    finally:
        parent_conn.send("done")
        process.join(timeout=60)
    assert process.exitcode == 0
```

Implementation notes for the executor (concrete debug order on failure):

1. If `create_cache_context` raises before construction, print the discovered
   format from `normalize_and_discover_per_layer_formats` on the unwrapped
   tensors — it must be `NL_X_TWO_X_NB_BS_HS`; if it is
   `NL_X_TWO_X_NB_BS_NH_HS`, the `LayoutHints` field name differs from
   `kv_layout` — read `lmcache/v1/gpu_connector/utils.py` and use the right
   field.
2. If store's byte comparison fails, dump `host.view(NL, NB * BS, HIDDEN)`
   against `_expected_staging()` per layer to see which plane ordering the
   fallback transfer produced before touching production code — the test
   mirrors the upstream `_transfer_per_layer_mla` ordering
   (`obj[:, offset:token_end]` on rank-3 staging).
3. `module.store` reads only `key.request_id` / `key.worker_id` (plus the
   faked `resolve_obj_keys`), so the SimpleNamespace key is sufficient.

- [ ] **Step 2: Run on the dev box**

Run: `cd /mnt/sdb/jjy/LMCache-Ascend && LMCACHEPATH=/mnt/sdb/jjy/LMCache timeout 600 python3 -m pytest tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py -v`
Expected: 3 PASS (`test_forced_mode_selects_lmcache_driven_context` plus the `npu:0` / `npu:1` parametrizations; the second skips on a single-device box). Follow the numbered debug order in the implementation notes on failure.

- [ ] **Step 3: Run the full plugin suite gate**

Run: `cd /mnt/sdb/jjy/LMCache-Ascend && LMCACHEPATH=/mnt/sdb/jjy/LMCache timeout 1800 python3 -m pytest tests/v1 -q`
Expected: baseline 552 passed / 32 failed / 10 skipped plus the new passing tests, no new failures (pre-existing failures are catalogued in project memory).

- [ ] **Step 4: Commit (in the plugin repo)**

```bash
git add tests/v1/multiprocess/test_npu_lmcache_driven_e2e.py
git commit -m "[MP] E2E test for LMCacheDriven store/retrieve on NPU

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Final verification (after all tasks)

- [ ] Upstream: `PYTHONPATH=/tmp/lmc_test_plugin python -m pytest tests/v1/platform/npu tests/v1/platform/test_cache_context_dispatch.py tests/v1/multiprocess/test_event_ipc_handle_path.py -q -p noshared` — all pass (device tests pass on the dev box, skip elsewhere).
- [ ] Upstream: `pre-commit run --files lmcache/v1/platform/npu/*.py tests/v1/platform/npu/*.py` (or `pre-commit run --all-files` if fast enough).
- [ ] Plugin: full-suite gate above.
- [ ] No changes outside `lmcache/v1/platform/npu/` (upstream) except new test files, the design doc, and the two plugin test files: `git diff --stat origin/dev...HEAD` after the work shows only those.

---

## Amendment (2026-09-02): Task 9 — B-lite upstream trio

Approved by the user after Task 8 falsified the Phase-1 "fallback covers all
formats" premise on 910B hardware. Three surgical, precedent-backed additions
to upstream shared code, then Task 8's E2E finishes against them.

### Task 9: NPU pointer views + MLA-tuple fallback transfer + planes-per-layer regroup

**Files:**
- Modify: `lmcache/v1/platform/torch_ops.py` (pointer view + transfer branch)
- Modify: `lmcache/v1/gpu_connector/utils.py` (`LayoutHints.planes_per_layer`, regroup in `normalize_and_discover_per_layer_formats`)
- Test: `tests/v1/platform/test_tensor_from_ptr.py` (extend or create), `tests/v1/platform/npu/test_npu_torch_ops_fallback.py`

**Interfaces:**
- Produces: `_tensor_from_npu_ptr(ptr, shape, dtype, device, total_bytes) -> torch.Tensor` (non-owning view, mirror of `_tensor_from_musa_ptr` using `torch._C._construct_storage_from_data_pointer`); `LayoutHints.planes_per_layer: int = 1`; regrouping of a flat per-layer tensor list into per-layer tuples when `planes_per_layer > 1`; a fallback dispatch branch for `EngineKVFormat.NL_X_TWO_X_NB_BS_HS` in `multi_layer_block_kv_transfer`.

Steps (TDD each):

1. `_tensor_from_npu_ptr`: add the NPU branch to `_tensor_from_ptr`'s dispatch
   (`if device.type == "npu": return _tensor_from_npu_ptr(...)`) and the
   helper, mirroring `_tensor_from_musa_ptr` line-for-line (same
   `torch._C._construct_storage_from_data_pointer` construction, same
   RuntimeError wrapping). Update the docstring's supported-device list.
   Test (NPU-gated, cpu-skipped): build a small `npu` tensor, view it through
   `_tensor_from_ptr(t.data_ptr(), t.shape, t.dtype, t.device)`, assert
   `view.data_ptr() == t.data_ptr()` and a mutation through the view is
   visible in `t`.

2. `LayoutHints.planes_per_layer: int = 1` (dataclass field, default keeps
   every existing caller unchanged). In
   `normalize_and_discover_per_layer_formats`, when `planes_per_layer > 1`
   and the discovered input is a flat list of tensors, regroup consecutive
   `planes_per_layer` tensors into per-layer tuples before format
   classification (engine-agnostic; detectors already classify tuple inputs).
   Test (CPU-runnable): flat list of `2L` planes with unequal widths +
   `LayoutHints(kv_layout="NHD", planes_per_layer=2)` classifies as
   `NL_X_TWO_X_NB_BS_HS`; `planes_per_layer=3` with three unequal widths
   classifies the same (DSA); default `1` leaves today's classification
   untouched (reuse the fixtures of
   `tests/v1/gpu_connector/test_kv_format_detection.py:179-205`).

3. Fallback branch for `NL_X_TWO_X_NB_BS_HS` in
   `torch_ops.multi_layer_block_kv_transfer`: new
   `_transfer_per_layer_mla_tuple(layer_planes, object_tensors, block_ids,
   n_block_ids, blocks_per_object, block_size, is_d2h,
   skip_prefix_n_blocks)` where `layer_planes[l]` is the tuple of that
   layer's `[NB, BS, W_p]` planes and the staging `object_tensors` are
   rank-3 `[L, tokens, sum(W_p)]`. Per chunk, per layer, per plane `p` with
   width `W_p` and slab offset `off_p = sum(W_q for q < p)`:
   D2H — `sel = torch.index_select(plane, 0, eff_idx)` (shape
   `[n_valid, BS, W_p]`), write
   `obj[l, token_lo:token_hi, off_p:off_p + W_p].copy_(sel.reshape(-1, W_p))`;
   H2D — the inverse `index_copy_` into each plane. Skip/valid-block logic
   mirrors `_transfer_per_layer_mla` (`_valid_block_range_indices`,
   `torch_ops.py:1557-1628`). `_normalize_paged_layers` must pass per-layer
   plane tuples through for this format. Test (CPU-runnable): roundtrip D2H→
   H2D with distinct values per (layer, plane, block) on CPU tensors through
   `device_ops`-level `multi_layer_block_kv_transfer`, staging buffer shaped
   `[L, tokens, Wl + Wr]`, asserting byte-exact both directions including
   `skip_prefix_n_blocks > 0`.

Commit: `[MP][Ascend] torch fallback: NPU pointer views, MLA-tuple transfer, plane regroup` (with trailer).

Global constraints: same as plan header; the three changes are the ONLY
shared-code edits sanctioned by the user's B-lite decision — anything beyond
them goes back to the user.
