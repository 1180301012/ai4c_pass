import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import triton_dispatch


# ---------------------------------------------------------------------------
# Pattern: single-arg — tmp_15_input (the (tmp_11+tmp_13)==0 result).
#
# By NOT including causal_mask in the pattern, we avoid the multi-use issue
# where the causal mask is referenced in different positions in the graph.
#
# tmp_15_input is a placeholder that binds to the eq result in the target.
# The getitem(tmp_10, [:,:,:,:N]) inside the pattern uses a concrete slice
# (same structure for bfloat16 / float16 / float32).
#
# Returns tmp_17 (the masked_fill result); downstream setitem/mul are
# unchanged and handled by torch.compile.
# ---------------------------------------------------------------------------
def pattern(tmp_15_input):
    tmp_16 = ((slice(None, None, None), slice(None, None, None),
               slice(None, None, None), slice(None, 21, None)))
    tmp_17 = tmp_16.masked_fill(tmp_15_input, -3.4028234663852886e+38)
    return tmp_17


def replacement_args(tmp_15_input):
    return (tmp_15_input, "from_pattern")


def replacement_func():
    return triton_dispatch