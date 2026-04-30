import torch
import triton
import triton.language as tl
from pass_dir.shared_routes import replacement_func


# Pattern matching function.
# Use a placeholder for the arange result and match the exact unsqueeze+repeat dataflow.
def pattern(tmp_0):
    tmp_1 = tmp_0.unsqueeze(0)
    tmp_2 = tmp_1.repeat(1, 1)
    return tmp_2


# Extract the input feeding the matched subgraph plus a shared dispatch route.
def replacement_args(tmp_0):
    return ("unsqueeze_repeat_1_1", tmp_0)


@triton.jit
def _copy_1d_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, vals, mask=mask)