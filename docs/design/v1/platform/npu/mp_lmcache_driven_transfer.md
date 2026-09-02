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
