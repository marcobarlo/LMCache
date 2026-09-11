# NPU Platform: Plane-Aggregating IPC Wrapper

`lmcache/v1/platform/npu/ipc_wrapper.py` ships worker KV caches across the
multiprocess wire on Ascend NPUs. It is the platform's implementation of
`DeviceSpec.ipc_wrapper_cls` (the class `resolve_kv_wrapper_factory` returns
for `device_type="npu"`), and it moved here from the LMCache-Ascend plugin's
`lmcache_ascend/v1/multiprocess/custom_types.py` so the binding needs no
plugin-side patch.

## Why it does not subclass `CudaIPCWrapper`

It subclasses `DeviceIPCWrapper` directly because Ascend differs from CUDA in
two mechanical ways:

| Aspect | CUDA | Ascend |
|---|---|---|
| Storage sharing | `UntypedStorage._share_cuda_()` / `_new_shared_cuda` | `_share_npu_()` / `_new_shared_npu` |
| Device identity | `get_device_properties(i).uuid` | none in the ACL runtime — derived from `npu-smi` (VDie ID, PCIe fallback) |
| Ordinal space | device count == addressable cards | logical ordinals can exceed physical cards; UUID discovery skips unprobeable ordinals with a warning |

## Multi-plane aggregation (the wire contract)

Engines may register a layer as **one tensor or a sequence of paged planes**:
vLLM-Ascend's model runner hands per-layer `(K, V)` pairs, MLA
`(latent, rope)` / DSA `(latent, rope, dsa)` tuples, arity-1 `(k,)` tuples,
and plane *lists* from `_adjust_kv_layout` / Mamba state tensors
(`vllm_ascend/worker/model_runner_v1.py::_reshape_kv_cache_tensors`).

The wrapper is **plane-aggregating**: one wrapper per registered layer value,
one `PlaneRecord` `(handle, dtype, shape, stride, storage_offset)` per plane
inside it. On the wire (`KVCache = list[DeviceIPCWrapper]`) the list element
count therefore equals the **layer** count, and `to_tensor()` reconstructs a
bare tensor for single-plane values or a tuple of planes otherwise. The
server-side `normalize_and_discover_per_layer_formats` therefore sees the
same per-layer tensor-or-tuple entries the engine registered.

This exercises the documented **multi-plane exception** of
`DeviceIPCWrapper` (see its class docstring): the singular interface fields
are not populated; equality compares `_plane_records` instead. `NpuIPCWrapper`
is currently the only implementation of the exception; generic code must not
assume `to_tensor()` returns a bare tensor without checking the device.

Worker-side format discovery (`create_engine_group_infos_from_vllm`) never
crosses the wire. It canonicalizes the engine dict with
`kv_wrap.per_layer_planes`: arity-1 sequences unwrap to a bare tensor;
larger sequences become tuples.

## Dispatch

`wrap_one_kv_cache` resolves the device from the value itself via
`gpu_connector.utils.get_device` (first-tensor descent), so a plane sequence
dispatches to its device's factory without any device-specific branch in
generic code. The import is function-local: the platform layer must not gain
a module-level dependency on the `gpu_connector` package (its `__init__`
pulls the heavy connector modules).

## Testing split

- CPU (upstream, `tests/v1/platform/npu/test_npu_ipc_wrapper.py`): registry
  binding, `wrap()` arity bookkeeping (storage sharing monkeypatched),
  equality/hash, pickle round-trip.
- Device (LMCache-Ascend plugin, spawn round-trips in
  `tests/v1/multiprocess/test_custom_types.py`): real `_share_npu_` /
  `_new_shared_npu` reconstruction, single- and multi-plane.
