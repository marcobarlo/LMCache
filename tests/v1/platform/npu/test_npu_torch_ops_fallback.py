# SPDX-License-Identifier: Apache-2.0
"""NPU-gated tests for the torch-fallback raw-pointer tensor views.

``_tensor_from_ptr`` must reconstruct a non-owning view over an Ascend NPU
device pointer so the fallback ``multi_layer_block_kv_transfer`` path can
serve NPU paged buffers. These tests need real Ascend hardware and skip
cleanly everywhere else.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import torch_ops

pytestmark = [
    pytest.mark.npu,
    pytest.mark.no_shared_allocator,
]

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU hardware is required",
)


@requires_npu
def test_tensor_from_ptr_npu_pointer_view() -> None:
    """A raw NPU pointer round-trips into an aliasing tensor view."""
    # Third Party
    import torch_npu  # noqa: F401

    original = torch.arange(24, dtype=torch.float32, device="npu").reshape(4, 6)
    view = torch_ops._tensor_from_ptr(
        original.data_ptr(), original.shape, original.dtype, original.device
    )

    assert view.data_ptr() == original.data_ptr()
    assert view.shape == original.shape
    assert view.dtype == original.dtype
    assert view.device.type == "npu"

    # The view must alias the original buffer, not copy it: a mutation
    # through the view has to be observed by the caller's tensor.
    view.fill_(7.0)
    assert torch.equal(original, torch.full_like(original, 7.0))
