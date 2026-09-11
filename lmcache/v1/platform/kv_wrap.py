# SPDX-License-Identifier: Apache-2.0
"""Device-agnostic helpers that wrap worker KV caches for IPC transport.

These helpers used to live under ``lmcache.integration.vllm`` for historical
reasons, but they are engine-neutral: dispatch happens purely via
:func:`resolve_kv_wrapper_factory` on the value's device type. Keeping them
here lets core transfer contexts (e.g. ``LMCacheDrivenTransferContext``) use
them without importing the vLLM integration package.

Wrapping is **per layer**: an engine may register a layer as one tensor or
as a sequence of paged planes, and each registered value becomes exactly one
IPC wrapper. On the receiving side ``to_tensor()`` reconstructs that same
value (a bare tensor or a plane tuple).
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.platform import resolve_kv_wrapper_factory

logger = init_logger(__name__)


def wrap_one_kv_cache(tensor: torch.Tensor) -> Any:
    """Dispatch on the value's device type via the platform registry.

    Concrete factories are supplied by the registered ``DeviceSpec`` objects,
    so this call site stays free of if/elif chains and external accelerators
    can provide their wrapper from an installed device-plugin wheel.

    The value may also be one layer's paged-plane sequence (e.g.
    vLLM-Ascend's per-layer ``(K, V)`` tuples): the device is probed by
    descending into the first tensor, and the whole sequence is handed to
    the device factory as one value. Whether plane sequences are supported
    is the device wrapper's decision (the plane-aggregating
    :class:`~lmcache.v1.platform.npu.ipc_wrapper.NpuIPCWrapper` accepts
    them; single-tensor wrappers reject them inside their ``__init__``).
    """
    # Local import: the platform layer must not gain a module-level
    # dependency on the gpu_connector package (its ``__init__`` drags in
    # the heavy connector modules and would risk an import cycle).
    # First Party
    from lmcache.v1.gpu_connector.utils import get_device

    return resolve_kv_wrapper_factory(get_device(tensor).type)(tensor)


def per_layer_planes(
    kv_caches: dict[str, "torch.Tensor | Sequence[torch.Tensor]"],
) -> list["torch.Tensor | tuple[torch.Tensor, ...]"]:
    """Canonicalize each layer to a bare tensor or a plane tuple.

    Args:
        kv_caches: Mapping from layer name to tensor or a per-layer
            sequence of tensors.

    Returns:
        One entry per layer in registration order. Arity-1 sequences
        unwrap to their only tensor; larger sequences become tuples.
    """
    canonical: list["torch.Tensor | tuple[torch.Tensor, ...]"] = []
    for value in kv_caches.values():
        if isinstance(value, torch.Tensor):
            canonical.append(value)
            continue
        if len(value) == 1:
            canonical.append(value[0])
            continue
        canonical.append(tuple(value))
    return canonical


def wrap_kv_caches(
    kv_caches: dict[str, "torch.Tensor | tuple[torch.Tensor, ...]"],
) -> KVCache:
    """Wrap every layer's KV cache for IPC transport.

    Args:
        kv_caches: Mapping from layer name to the layer's KV tensor or
            per-layer plane sequence (e.g. vLLM-Ascend's (K, V) pairs).
            Each value becomes exactly one wrapper: one list element per
            layer, reconstructing to a bare tensor or a plane tuple.

    Returns:
        The list of per-layer IPC wrappers, ready for the msgspec wire.
    """
    values = list(kv_caches.values())
    # Emit a per-layer shape/dtype structure summary (shared walker, see
    # get_shape_and_dtype) so the operator can verify the exact tensor
    # geometry being shipped to the LMCache server, then the low-noise
    # count of handles being wrapped.
    # First Party
    from lmcache.v1.gpu_connector.utils import get_shape_and_dtype

    # Plane sequences flow unannotated by design (see wrap_one_kv_cache).
    structures = get_shape_and_dtype(values)  # type: ignore[arg-type]
    logger.debug(
        "KV cache transfer keeping %d layer(s) (shape, dtype):\n%s",
        len(structures),
        "\n".join(f"  [{i}]  {s}" for i, s in enumerate(structures)),
    )
    logger.info("Wrapping %d KV cache layers for IPC", len(values))
    # Per-iteration resource management: if wrapping the N-th layer
    # raises, ``shm_unlink`` whatever earlier iterations already
    # registered with POSIX SHM so the named segments do not outlive
    # the failed batch. CUDA/NPU wrappers do not own a named segment
    # and are skipped via the duck-typed ``shm_name`` check.
    wrappers: KVCache = []
    try:
        for value in values:
            wrappers.append(wrap_one_kv_cache(value))  # type: ignore[arg-type]
    except BaseException:
        _release_partial_kv_wrappers(wrappers)
        raise
    return wrappers


def _release_partial_kv_wrappers(wrappers: list[Any]) -> None:
    """Best-effort unlink of SHM segments owned by partially built wrappers.

    Used by :func:`wrap_kv_caches` to roll back a half-finished batch
    when a later iteration raises. Only POSIX-SHM-backed wrappers carry
    a ``shm_name`` attribute, so other wrapper kinds (e.g. CUDA/NPU IPC)
    are silently skipped.
    """
    # First Party
    from lmcache.v1.multiprocess.posix_shm import shm_unlink

    for w in wrappers:
        name = getattr(w, "shm_name", None)
        if name is None:
            continue
        try:
            shm_unlink(name)
        except Exception:  # pragma: no cover - best effort
            logger.debug("shm_unlink failed during rollback", exc_info=True)
