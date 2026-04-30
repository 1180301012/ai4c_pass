import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import triton_dispatch


# ---------------------------------------------------------------------------
# Pattern for float16 / float32 graphs (ATen-level ops).
# These models use aten.masked_fill.Scalar in the compiled graph.
# tmp_15_input is a placeholder for the (tmp_11+tmp_13)==0 result.
# ---------------------------------------------------------------------------
def pattern(causal_mask, tmp_15_input):
    tmp_16 = causal_mask[(slice(None, None, None), slice(None, None, None),
                           slice(None, None, None), slice(None, 21, None))]
    tmp_17 = torch.ops.aten.masked_fill.Scalar(tmp_16, tmp_15_input, -3.4028234663852886e+38)
    return tmp_17


def replacement_args(causal_mask, tmp_15_input):
    # Append route="aten" so the shared dispatch knows which kernel to use
    return (causal_mask, tmp_15_input, "aten")


def replacement_func():
    return triton_dispatch