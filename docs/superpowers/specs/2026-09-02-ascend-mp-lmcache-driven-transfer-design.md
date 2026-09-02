# Ascend MP Mode: LMCacheDriven Transfer Path (Phase 1) — Design

- Date: 2026-09-02
- Branch: `support_ascend_mp_mode_dev_3` (LMCache) / `support_ascend_mp_mode_dev_3` (LMCache-Ascend)
- Status: approved design, pending implementation

## Problem

LMCache MP mode offers two worker↔server transfer paths. On Ascend NPU only
`engine_driven` works today (the worker copies through SHM/pickle; the
LMCache-Ascend plugin's `npu_gather.py` overrides it with a fused
`c_ops.multi_layer_kv_transfer` kernel). The `lmcache_driven` path — where the
worker registers its KV cache once via IPC handles and the MPCacheServer drives
D2H/H2D transfers directly against the worker's device memory — fails at
registration because three platform capabilities are missing:

1. `NpuDeviceSpec.event_ipc_backend` is unset → `get_event_ipc_backend()`
   raises at `register_kv_cache` (`lmcache_driven_transfer.py`).
2. `NpuDeviceSpec.create_cache_context` is unset → base raises
   `NotImplementedError` (no NPU `BaseCacheContext` exists).
3. The stream-ordered completion recorder falls back to immediate enqueue,
   letting `finish_write` commit storage locks before async NPU copies drain
   (a concurrent RETRIEVE could read torn data).

## Verified enablers (910B, CANN 9.1.0, torch_npu 2.10.0.post4)

- `torch.npu.Event` supports the CUDA-style interprocess ABI:
  `Event(interprocess=True)`, `ipc_handle()`, `Event.from_ipc_handle(device,
  handle)`, `record(stream)`, `wait(stream)`, `query()`, `synchronize()`.
  Cross-process import + synchronize verified with a two-process probe.
- `torch.npu.Stream.npu_stream` exposes the raw stream pointer
  (`torch.npu.ExternalStream` does NOT exist; there is no raw-pointer stream
  wrapper in torch_npu).
- The plugin's `AscendIPCWrapper` (`_share_npu_` / `_new_shared_npu`) is
  already registered on `NpuDeviceSpec.ipc_wrapper_cls` by monkey-patch.
- The plugin's `LazyMemoryAllocator` patch backs the server pinned pool with
  `aclrtMallocHost` — required by the D2H/H2D staging copies.
- `torch_ops.multi_layer_block_kv_transfer` (pure-torch fallback) covers all
  engine KV formats, so Phase 1 needs no new kernels.
- CANN 9.1 `acl.rt` python bindings expose `synchronize_stream` and
  `launch_callback` (the latter reserved for Phase 2).

## Decisions (user-approved)

1. **Correctness first, kernels later.** Phase 1 is pure Python, no csrc
   changes. Phase 2 (separate PRs) adds native `aclrtLaunchCallback`-based
   completion recording and native block-transfer kernels in the plugin.
2. **Explicit enablement.** NPU deployments opt in via
   `LMCACHE_MP_TRANSFER_MODE=lmcache_driven`. The AUTO route
   (`worker_transfer.py:944`) keeps sending `npu` to engine-driven; flipping
   AUTO is a later, separate PR after burn-in.
3. **No modification to `GPUCacheContext`.** `NpuCacheContext` subclasses
   `BaseCacheContext` directly in a self-contained file. Zero edits to shared
   code: all upstream changes are additive files under `platform/npu/` plus
   three `NpuDeviceSpec` members.
4. **Mirror the MUSA platform** (`lmcache/v1/platform/musa/`), the existing
   precedent for a CuPy-less / no-native-host-callback platform on the
   LMCacheDriven path: same file layout, same structural patterns
   (host-callback stream adapter, flat temp buffer with two offset maps,
   sync-then-enqueue completion recording, cached lazy spec properties).
5. **Docstring policy:** CUDA/MUSA style — module and class one-liners, full
   `Args`/`Returns`/`Raises` on public API, none on trivial private helpers.

## Design

### Data flow (store; retrieve is symmetric with H2D)

```
worker                                   MPCacheServer
------                                   -------------
KV tensors ──AscendIPCWrapper──► register → NpuCacheContext (imports mappings)
producer event ──ipc handle──► ZMQ ──► import_event + wait on transfer stream
                                         paged KV ──torch fallback block
                                           transfer──► NPU staging buffer
                                         staging ──async D2H──► pinned MemoryObj
completion event ◄──ipc handle── ZMQ ◄── record + export
                                         finish_write: stream sync, then
                                           immediate enqueue (MUSA pattern)
```

### Upstream: `lmcache/v1/platform/npu/`

#### `event_ipc.py` (new, ~60 lines)

`NpuEventIPCBackend(DefaultEventIPCBackend)` with `device_type = "npu"`.
Resolves the `torch.npu` module lazily (mirrors how `pin_memory.py` guards
CANN imports) and delegates to the inherited default backend. The inherited
`check_event_support` fails closed on old torch_npu builds that lack
`from_ipc_handle` — no extra gating layer needed (unlike MUSA, whose ABI needs
an availability probe).

#### `cache_context.py` (new, ~350 lines)

Self-contained, structurally mirroring `MUSACacheContext`:

- `unwrap_kv_cache_tensors(kv_caches)` — `[w.to_tensor() for w in kv_caches]`.
- `_NpuHostCallbackStream` — adapter exposing `.ptr` (→ `npu_stream`) for the
  shared `record_*_on_stream(stream.ptr, ...)` call sites, plus
  `launch_host_func` falling back to synchronize-then-run (MUSA semantics).
- `_TempNpuBuffer` — one flat `uint8` staging tensor sized
  `single_batch_bytes * max_batch_size(=4)`, two offset maps
  `(batch, kernel_group)` and `(batch, object_group)`; typed views per kernel
  group; kernel-group contiguity inside each object group preserved (the
  staging memcpy contract). The GDS-driven third offset map is omitted.
- `NpuCacheContext(BaseCacheContext)`, `device_type = "npu"`:
  `__init__`/`_initialize` with wrapper rollback on failure; format discovery
  via shared `normalize_and_discover_per_layer_formats`; `KVLayerGroupsManager`;
  1M-entry block-ids buffer; per-kernel-group int64 pointer tables
  (`get_group_data_ptrs`); `torch_dev.Stream`; close() = stream sync +
  close wrappers. Properties: `stream`, `cupy_stream` (the adapter),
  `get_kernel_group_kv_pointers`, `get_temp_kernel_group_buffer`,
  `get_temp_object_group_buffer`, `max_batch_size`,
  `get_kernel_group_shape_dtype`, `cache_size_per_token`.

#### `device_ops.py` (extend, +~50 lines)

`NpuDeviceOps` overrides `record_completion_on_stream` and
`record_event_on_stream` (MUSA `device_ops.py:559-616` pattern):
synchronize the stream, then call the torch-baseline immediate enqueue.

`_synchronize_npu_stream_pointer(ptr)` prefers
`acl.rt.synchronize_stream(ptr)` (guarded import, reusing the loader pattern
from `pin_memory.py`); fallback: verify `torch.npu.current_stream()
.npu_stream == ptr` and synchronize the current stream (the LMCacheDriven call
sites always run inside `torch_dev.stream(cache_context.stream)`).

Rationale: torch_npu lacks a host-callback primitive; synchronizing at submit
preserves the storage-ownership contract (`finish_write` must observe drained
copies) at the cost of blocking the AFFINITY handler thread until copies
finish. Phase 2's native `aclrtLaunchCallback` implementation, exported from
`lmcache_ascend.c_ops`, shadows these Python overrides automatically via
`DeviceOps.bind_native` — no upstream change needed then.

#### `__init__.py` (extend, +~25 lines)

`NpuDeviceSpec` gains (all lazily imported / cached, MUSA style):
`event_ipc_backend` property → cached `NpuEventIPCBackend()`; 
`create_cache_context(*args, **kwargs)` → `NpuCacheContext(...)`.
`ipc_wrapper_cls` stays plugin-side (existing monkey-patch registers
`AscendIPCWrapper`). `is_handle_transfer_available` keeps the default True.

### Plugin: LMCache-Ascend

No functional changes in Phase 1. Test enablement only:

- Selectively un-ignore `tests/v1/multiprocess/` collection (at minimum
  `test_custom_types.py`, whose cross-process `AscendIPCWrapper` spawn test is
  the IPC leg of this path).
- New E2E test on 910B: two processes (worker registers wrappers + events;
  server-side `LMCacheDrivenTransferModule` store/retrieve) with byte-level
  roundtrip verification, including a worker on a non-zero NPU device.
- Run recipe per project memory: `LMCACHEPATH=/mnt/sdb/jjy/LMCache`, pytest
  plugin marking all items `no_shared_allocator`.

### Error handling

- Registration failure rolls back partially imported IPC mappings
  (constructor try/except, MUSA/GPU precedent).
- Old torch_npu: `check_event_support` raises at REGISTER with a clear
  message; deployment falls back to engine-driven by unsetting the env var.
- Unregistered-instance STORE/RETRIEVE already returns `(b"", False)` —
  untouched.
- Ordering failure of `acl.rt.synchronize_stream` surfaces as the handler
  exception path already covered by store/retrieve's try/finally.

### Testing

- Upstream, CPU-runnable: `NpuHostCallbackStream.ptr`; sync-then-enqueue
  ordering with a fake event/stream; spec property wiring
  (`event_ipc_backend`, `create_cache_context` lazy resolution).
- Upstream, NPU-gated: event backend cross-process roundtrip;
  `NpuCacheContext` layout parity (shapes/offsets) against the same geometry
  as the GPU context on identical layer specs.
- Plugin: the E2E above.

## Open items to resolve during implementation

1. **MLA staging shape.** MUSA's `_TempMUSABuffer._get_shape_for_kernel_group`
   special-cases `NL_X_NB_BS_HS` to 3D `[L, slots, W]`. The NPU MLA tuple
   format `NL_X_TWO_X_NB_BS_HS` (branch top commit) must be checked against
   what the torch-fallback transfer (`_transfer_per_layer_mla`) actually
   writes/reads before fixing 3D-vs-4D in `_TempNpuBuffer`.
2. **Multi-NPU semantics.** Cross-process event import was probed on npu:0
   only; the E2E test must cover a worker on a non-zero device.
3. **Cross-thread `Event.query()` / stream sync from the dispatcher-driven
   paths** — verified implicitly by the E2E test.

## Phase 2 (out of scope; separate PRs)

- Plugin c_ops: native `record_completion_on_stream` via `aclrtLaunchCallback`
  (removes the submit-time sync); native `multi_layer_block_kv_transfer` /
  `execute_object_group_transfer` reusing the fused kernel machinery.
- Flip AUTO routing for `npu` after burn-in.
- `blend.py`'s CuPy dependency means CPU-offload attention stays unsupported
  on NPU; it must raise clearly if configured.
