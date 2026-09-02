# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU ops backend: bulk-bind the plugin's ``lmcache_ascend.c_ops``.

:class:`NpuDeviceOps` calls :meth:`bind_native` in :meth:`ensure_native`
to layer the external Ascend plugin's compiled extension on top of the
torch baseline. All NPU-specific kernels live in the ``lmcache_ascend``
plugin; this package only wires detection and backend selection. If the
plugin is missing, a warning is logged and the instance stays on the
torch fallback (soft-fail, same as CUDA).
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.base.device_ops import DeviceOps

logger = init_logger(__name__)


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
    # Some CANN binding versions return a NumPy scalar rather than a plain
    # int; coerce so a non-zero error code is never mistaken for success.
    if ret is not None and int(ret) != 0:
        raise RuntimeError(
            f"aclrtSynchronizeStream failed with error {ret} for stream {stream_ptr}"
        )


class NpuDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "npu"

    def ensure_native(self) -> None:
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # Third Party
            import lmcache_ascend.c_ops as native
        except ImportError:
            logger.warning(
                "lmcache_ascend.c_ops plugin not found; NpuDeviceOps stays "
                "on the torch baseline for all ops."
            )
            return
        self.bind_native(native)
        # The plugin's c_ops re-exports torch-fallback symbols the compiled
        # module lacks; instance-binding those would shadow the class-level
        # stream-ordered overrides below (the storage ownership contract).
        # Drop re-exports only: a genuine native implementation stays bound.
        for name in ("record_completion_on_stream", "record_event_on_stream"):
            bound = self.__dict__.get(name)
            if bound is not None and bound is getattr(torch_ops, name, None):
                del self.__dict__[name]

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
        callback backend: :meth:`ensure_native` keeps a genuine native
        binding of this name in place while dropping torch-fallback
        re-exports.

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
