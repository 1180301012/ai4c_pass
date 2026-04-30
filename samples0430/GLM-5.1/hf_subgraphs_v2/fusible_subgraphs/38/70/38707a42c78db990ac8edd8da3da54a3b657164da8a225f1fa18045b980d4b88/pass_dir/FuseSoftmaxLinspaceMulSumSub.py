import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - mirrors the exact computation in model.py
def pattern(in_0):
    tmp_0 = torch.nn.functional.softmax(in_0, dim = 1)
    tmp_1 = torch.linspace(0, 4, steps = 5, device = device(type='cuda', index=0))
    tmp_2 = tmp_0 * tmp_1
    tmp_3 = tmp_2.sum(dim = 1)
    tmp_4 = 5 - tmp_3
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Fused Triton kernel that computes:
# softmax(x, dim=1) * linspace(0,4,5) -> sum(dim=1) -> 5 - result
# All in float32 for numerical stability
@triton.jit
def fused_softmax_weighted_sum_kernel(
    in_ptr,
    out_ptr,
    num_rows,
    NUM_COLS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, NUM_COLS)
    
    # Load input row and convert to float32 for numerical stability
    x = tl.load(in_ptr + row_idx * NUM_COLS + offsets).to(tl.float32)
    
    # Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    denom = tl.sum(x_exp, axis=0)
    softmax_out = x_exp / denom
    
    # Multiply by linspace values [0, 1, 2, 3, 4]
    # Hardcode linspace to avoid creating and loading a constant tensor
    linspace_vals = offsets.to(tl.float32)
    weighted = softmax_out * linspace_vals
    
    # Sum along the row
    result = tl.sum(weighted, axis=0)
    
    # 5 - result
    out = 5.0 - result
    
    # Store output
    tl.store(out_ptr + row_idx, out)

# Kernel wrapper - must be decorated with @torch.fx.wrap
@torch.fx.wrap
def kernel_wrapper(in_0):
    num_rows = in_0.shape[0]
    # Output is float32 due to type promotion (bfloat16/float16 * float32 -> float32)
    out = torch.empty(num_rows, dtype=torch.float32, device=in_0.device)
    
    grid = (num_rows,)
    fused_softmax_weighted_sum_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        num_rows=num_rows,
        NUM_COLS=5,  # Fixed because pattern matches linspace(0,4,steps=5)
    )
    
    return (out,)

# Replacement function - returns the function reference (NOT a call)
def replacement_func():
    return kernel_wrapper