# SPDX-License-Identifier: Apache-2.0
"""Device-agnostic helpers that wrap worker KV caches for IPC transport.

These helpers used to live under ``lmcache.integration.vllm`` for historical
reasons, but they are engine-neutral: dispatch happens purely via
:func:`resolve_kv_wrapper_factory` on ``tensor.device.type``. Keeping them
here lets core transfer contexts (e.g. ``LMCacheDrivenTransferContext``) use
them without importing the vLLM integration package.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.platform import resolve_kv_wrapper_factory

logger = init_logger(__name__)


def wrap_one_kv_cache(tensor: torch.Tensor) -> Any:
    """Dispatch by ``tensor.device.type`` via the platform registry.

    Concrete factories are supplied by the registered ``DeviceSpec`` objects,
    so this call site stays free of if/elif chains and external accelerators
    can provide their wrapper from an installed device-plugin wheel.
    """
    return resolve_kv_wrapper_factory(tensor.device.type)(tensor)


def flatten_kv_cache_values(
    kv_caches: dict[str, "torch.Tensor | tuple[torch.Tensor, ...]"],
) -> list[torch.Tensor]:
    """Flatten per-layer tensor-or-tuple values into one ordered list.

    Args:
        kv_caches: Mapping from layer name to the layer's KV tensor or, for
            engines that hand per-layer plane tuples (e.g. vLLM-Ascend's
            per-layer (K, V) pairs), the tuple of that layer's planes.

    Returns:
        Every tensor in layer-then-plane order.
    """
    flat: list[torch.Tensor] = []
    for value in kv_caches.values():
        if isinstance(value, (tuple, list)):
            flat.extend(value)
        else:
            flat.append(value)
    return flat


def planes_per_layer(
    kv_caches: dict[str, "torch.Tensor | tuple[torch.Tensor, ...]"],
) -> int:
    """Return the uniform per-layer plane arity of ``kv_caches``.

    Args:
        kv_caches: Mapping from layer name to tensor or per-layer tuple.

    Returns:
        The tuple arity when every layer value is a tuple of the same
        arity greater than one; otherwise ``1``. Mixed or arity-1 layouts
        return ``1`` so the server-side detection surfaces the structure
        it actually receives instead of guessing.
    """
    if not kv_caches:
        return 1
    values = list(kv_caches.values())
    if not all(isinstance(value, (tuple, list)) for value in values):
        return 1
    arities = {len(value) for value in values}
    if len(arities) != 1:
        return 1
    arity = arities.pop()
    return arity if arity > 1 else 1


def wrap_kv_caches(
    kv_caches: dict[str, "torch.Tensor | tuple[torch.Tensor, ...]"],
) -> KVCache:
    """Wrap every KV cache tensor for IPC transport.

    Args:
        kv_caches: Mapping from layer name to the layer's KV tensor or
            per-layer plane tuple (e.g. vLLM-Ascend's (K, V) pairs); tuple
            values are flattened in layer-then-plane order, so pair this
            with ``LayoutHints.planes_per_layer`` so the server regroups
            the flat wrapper list back into layers.

    Returns:
        The list of per-tensor IPC wrappers, ready for the msgspec wire.
    """
    flat = flatten_kv_cache_values(kv_caches)
    # Emit a per-tensor (shape, dtype) summary so the operator can verify
    # the exact tensor geometry being shipped to the LMCache server, then
    # the low-noise count of handles being wrapped.
    kept_summary = [(tuple(tensor.shape), str(tensor.dtype)) for tensor in flat]
    logger.debug(
        "KV cache transfer keeping %d tensor(s) (shape, dtype):\n%s",
        len(kept_summary),
        "\n".join(
            f"  [{i}]  shape={shape}  dtype={dtype}"
            for i, (shape, dtype) in enumerate(kept_summary)
        ),
    )
    logger.info("Wrapping %d KV cache tensors for IPC", len(flat))
    # Per-iteration resource management: if wrapping the N-th tensor
    # raises, ``shm_unlink`` whatever earlier iterations already
    # registered with POSIX SHM so the named segments do not outlive
    # the failed batch. CUDA wrappers do not own a named segment and
    # are skipped via the duck-typed ``shm_name`` check.
    wrappers: KVCache = []
    try:
        for tensor in flat:
            wrappers.append(wrap_one_kv_cache(tensor))
    except BaseException:
        _release_partial_kv_wrappers(wrappers)
        raise
    return wrappers


def _release_partial_kv_wrappers(wrappers: list[Any]) -> None:
    """Best-effort unlink of SHM segments owned by partially built wrappers.

    Used by :func:`wrap_kv_caches` to roll back a half-finished batch
    when a later iteration raises. Only POSIX-SHM-backed wrappers carry
    a ``shm_name`` attribute, so other wrapper kinds (e.g. CUDA-IPC)
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
