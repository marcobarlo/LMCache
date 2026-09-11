# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU IPC wrapper implementation.

:class:`NpuIPCWrapper` shares KV-cache tensors across processes through
torch_npu's storage IPC (``_share_npu_`` / ``_new_shared_npu``), mirroring
what :class:`~lmcache.v1.platform.cuda.ipc_wrapper.CudaIPCWrapper` does for
CUDA. Two Ascend specifics force it to subclass
:class:`~lmcache.v1.platform.base.ipc_wrapper.DeviceIPCWrapper` directly
instead of riding the CUDA wrapper:

* the ACL runtime has no device UUID, so device identity comes from
  ``npu-smi`` (VDie / PCIe ID) instead of ``get_device_properties``;
* logical device ordinals can outnumber the physical cards ``npu-smi``
  addresses, so UUID discovery tolerates per-ordinal probe failures.

The wrapper is **plane-aggregating**: engines may register a layer as one
tensor or as a sequence of paged planes (vLLM-Ascend hands per-layer
``(K, V)`` pairs and MLA/DSA ``(latent, rope[, dsa][, scale])`` tuples).
``wrap`` accepts either and :meth:`to_tensor` reconstructs the same shape --
a bare tensor for single-plane values, a tuple for multi-plane ones -- so
server-side format detection sees the registered per-layer structure
directly.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import Any, ClassVar

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

logger = init_logger(__name__)

#: Per-plane IPC record: ``(handle, dtype, shape, stride, storage_offset)``.
#: ``handle`` is the opaque payload of ``UntypedStorage._share_npu_()``;
#: everything else mirrors the base-class interface fields, per plane.
PlaneRecord = tuple[Any, torch.dtype, tuple[int, ...], tuple[int, ...], int]


def _torch_npu() -> Any:
    """Return ``torch.npu``, failing closed when torch_npu is absent."""
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError(
            "torch_npu is not installed; NPU KV-cache IPC is unavailable."
        )
    return npu


class NpuIPCWrapper(DeviceIPCWrapper):
    """Plane-aggregating IPC wrapper for Ascend NPU KV tensors.

    This class exercises the documented multi-plane exception of
    :class:`DeviceIPCWrapper`: it does not populate the base-class
    singular fields (``handle`` / ``dtype`` / ``shape`` / ``stride`` /
    ``storage_offset``) but keeps one :data:`PlaneRecord` per plane in
    ``_plane_records`` instead, and :meth:`to_tensor` returns a tuple of
    tensors for multi-plane values. Only the NPU wrapper implements the
    exception today.
    """

    #: ``torch.device.type`` this wrapper handles; also read by the
    #: server's ``_detect_device_type`` to route to the NPU device spec.
    device_type: ClassVar[str] = "npu"

    #: Per-plane ``(handle, dtype, shape, stride, storage_offset)`` records.
    _plane_records: tuple[PlaneRecord, ...]

    @classmethod
    def wrap(cls, value: "torch.Tensor | Sequence[torch.Tensor]") -> "NpuIPCWrapper":
        """Factory used by :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            value: A single KV tensor, or one layer's paged planes as a
                sequence (e.g. vLLM-Ascend's ``(K, V)`` / ``(latent, rope)``
                tuples). Plane order is preserved across the wire.

        Returns:
            A new :class:`NpuIPCWrapper` aggregating ``value``'s planes for
            the multiprocess wire.
        """
        return cls(value)

    def __init__(self, value: "torch.Tensor | Sequence[torch.Tensor]") -> None:
        """Share every plane of ``value`` for cross-process reconstruction.

        Args:
            value: A single KV tensor or one layer's plane sequence. All
                planes must live on the same NPU device.

        Raises:
            RuntimeError: If a plane's storage cannot be shared through
                torch_npu storage IPC.
        """
        planes: list[torch.Tensor] = (
            [value] if isinstance(value, torch.Tensor) else list(value)
        )
        if not planes:
            raise ValueError(
                "NpuIPCWrapper requires at least one plane, got an empty sequence."
            )
        records: list[PlaneRecord] = []
        for plane in planes:
            storage = plane.untyped_storage()
            handle = storage._share_npu_()  # type: ignore[attr-defined]  # noqa: SLF001
            records.append(
                (
                    handle,
                    plane.dtype,
                    tuple(plane.shape),
                    tuple(plane.stride()),
                    int(plane.storage_offset()),
                )
            )
        self._plane_records = tuple(records)

        # Device identity is shared by every plane of the layer; probe the
        # first one. Base-class singular fields stay unset (see class
        # docstring: the documented multi-plane exception).
        device_index = planes[0].device.index
        self.device_uuid = self._get_device_uuid(device_index)

    def to_tensor(self) -> "torch.Tensor | tuple[torch.Tensor, ...]":  # type: ignore[override]
        """Reconstruct the wrapped layer in this process.

        Note:
            ``torch.npu`` must be initialized before this function is
            called (guarded by ``torch_dev.init()`` at the call sites).

        Returns:
            The bare tensor for a single-plane wrapper, otherwise the
            tuple of the layer's plane tensors in registration order.
        """
        device_index = self._get_device_index_from_uuid(self.device_uuid)
        tensors: list[torch.Tensor] = []
        for handle, dtype, shape, stride, storage_offset in self._plane_records:
            storage = torch.UntypedStorage._new_shared_npu(  # type: ignore[attr-defined]  # noqa: SLF001
                device_index, *handle[1:]
            )
            t = torch.empty((), device=device_index, dtype=dtype)
            t.set_(storage, storage_offset, shape, stride)
            tensors.append(t)
        return tensors[0] if len(tensors) == 1 else tuple(tensors)

    def __eq__(self, other: object) -> bool:
        # Base-class equality compares the singular interface fields this
        # wrapper does not populate; compare the per-plane records instead.
        if not isinstance(other, NpuIPCWrapper):
            return False
        return (
            self._plane_records == other._plane_records
            and self.device_uuid == other.device_uuid
        )

    def __hash__(self) -> int:
        return hash((self._plane_records, self.device_uuid))

    @staticmethod
    def _get_device_uuid(device_index: int) -> str:
        """Return a stable identity string for NPU device ``device_index``.

        The ACL runtime exposes no UUID, so identity is derived from
        ``npu-smi``: the VDie (silicon) ID when present, the PCIe bus ID
        as fallback, and ``<device name>-<ordinal>`` as a last resort
        (not guaranteed unique across hosts).

        Args:
            device_index: Logical NPU ordinal.

        Returns:
            The device identity string.

        Raises:
            RuntimeError: If the ``npu-smi`` probe fails.
        """
        # Standard
        import re
        import subprocess

        device_name = _torch_npu().get_device_name()

        try:
            # Run the npu-smi command
            cmd = [
                "npu-smi",
                "info",
                "-t",
                "board",
                "-i",
                str(device_index),
                "-c",
                "0",
            ]
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode(
                "utf-8"
            )

            # 1. Try to find VDie ID
            # Matches: "VDie ID : XXXXX XXXX..."
            vdie_match = re.search(r"VDie ID\s*:\s*([0-9A-F ]+)", result)
            if vdie_match:
                raw_id = vdie_match.group(1).replace(" ", "")
                if raw_id and not all(c == "0" for c in raw_id):
                    return f"{device_name}-{raw_id}"

            # 2. Fallback to PCIe Bus Info (Best Local ID)
            # Matches: "PCIe Bus Info : 0000:C1:00.0"
            pci_match = re.search(r"PCIe Bus Info\s*:\s*([0-9A-Fa-f:.]+)", result)
            if pci_match:
                return f"{device_name}-{pci_match.group(1)}"

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("Failed to retrieve device UUID from npu-smi.") from e

        # 3. Final fallback (unlikely to be unique across hosts)
        return f"{device_name}-{device_index}"

    @classmethod
    def _discover_devices(cls) -> None:
        """Map every probeable NPU's identity string to its ordinal.

        ``torch.npu.device_count()`` reports LOGICAL devices while
        ``npu-smi`` addresses PHYSICAL card IDs, so the two ranges can
        diverge (e.g. 16 logical devices on 8 cards): ordinals past the
        last card make the probe fail. Those are skipped with a warning
        instead of aborting UUID discovery on the whole host.
        """
        if not torch_dev.is_available():
            return
        num_devices = torch_dev.device_count()
        with DeviceIPCWrapper._device_mapping_lock:
            if DeviceIPCWrapper._discovered_device_mapping:
                return  # Already discovered

            for i in range(num_devices):
                try:
                    device_uuid = cls._get_device_uuid(i)
                except RuntimeError:
                    logger.warning(
                        "Skipping NPU device ordinal %d during UUID "
                        "discovery: npu-smi could not resolve it (logical "
                        "device count exceeds the physical card IDs).",
                        i,
                    )
                    continue
                DeviceIPCWrapper._discovered_device_mapping[device_uuid] = i
