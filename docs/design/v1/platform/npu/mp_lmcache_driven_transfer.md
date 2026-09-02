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
| `cache_context.py` | `NpuCacheContext` (subclass of `BaseCacheContext`) — imports worker KV mappings from `AscendIPCWrapper` and owns `_TempNpuBuffer` staging and the transfer stream. `get_kernel_group_kv_pointers` returns per-layer plane **views** — the zero-copy IPC-imported tensors themselves, or tuples of plane tensors for `NL_X_TWO_X_NB_BS_HS` — not a device-resident pointer table: per-plane widths are unrecoverable from the summed `shape_desc.hs`, and the torch fallback consumes per-layer structures directly. The Phase-2 native fast path must revisit this method if it needs a real pointer table. `_NpuHostCallbackStream` adapts the torch_npu stream to the `cupy_stream` contract (`.ptr` from `npu_stream`; `launch_host_func` degrades to synchronize-then-run). |
| `device_ops.py` | `NpuDeviceOps` binds `lmcache_ascend.c_ops` and keeps completion/event recording stream-ordered: `_synchronize_npu_stream_pointer` (via `acl.rt.synchronize_stream`) runs before the immediate-enqueue fallback, preserving the `finish_write` storage-ownership contract until the plugin ships a native `aclrtLaunchCallback` recorder. |

## Shared-code additions (B-lite)

Three touches outside `lmcache/v1/platform/npu/` are sanctioned for this
path; all other NPU-specific code lives in the plugin or under `npu/`:

- `torch_ops._tensor_from_npu_ptr` (plus its `npu` dispatch arm in
  `_tensor_from_ptr`) and the `NL_X_TWO_X_NB_BS_HS` branch of the
  `multi_layer_block_kv_transfer` fallback
  (`_transfer_per_layer_mla_tuple`) let the torch fallback reconstruct NPU
  tensors from raw pointers and transfer per-layer plane tuples.
- `LayoutHints.planes_per_layer` and the flat-list regroup in
  `normalize_and_discover_per_layer_formats` classify vLLM-Ascend's flat
  per-plane registration list as per-layer `(latent, rope[, dsa])` tuples
  before format detection.
- `_normalize_lmcache_objects` honors an explicit `device` for pointer-mode
  inputs, so reconstructed object chunk views alias device-resident staging
  buffers instead of defaulting to CPU.

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

MLA/DSA tuple layouts additionally require the worker's registration
`LayoutHints` to carry `planes_per_layer` (with `kv_layout="NHD"`).
Today only tests set that hint: engine-side emission (the vLLM-Ascend
connector) is enablement work deferred together with the AUTO flip.
Until it lands, a production MLA worker opting into `lmcache_driven`
without the hint will have its flat plane list classified as the
per-layer (K, V) format, producing a wrong object layout.

Spec: `docs/superpowers/specs/2026-09-02-ascend-mp-lmcache-driven-transfer-design.md`.
