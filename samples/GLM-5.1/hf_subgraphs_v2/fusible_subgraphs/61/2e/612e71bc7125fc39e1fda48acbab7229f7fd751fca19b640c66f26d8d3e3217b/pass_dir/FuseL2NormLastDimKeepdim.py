import torch
import triton
import triton.language as tl

# Pattern matching function - matches L2 normalization: norm(p=2, dim=-1, keepdim=True) + divide
def pattern(in_1):
    tmp_0 = in_1.norm(p = 2, dim = -1, keepdim = True)
    tmp_1 = in_1 / tmp_0
    return tmp_1

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Triton kernel for L2 normalization fused (norm + div)
@triton.jit
def l2_norm_kernel(
    X_ptr, OUT_ptr,
    nrows, ncols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    if row_idx >= nrows:
        return
    
    row_start = row_idx * ncols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < ncols
    
    # Load the row
    x = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)
    
    # Compute sum of squares (in float32 for precision)
    x_float = x.to(tl.float32)
    sum_sq = tl.sum(x_float * x_float, axis=0)
    
    # Compute L2 norm
    norm_val = tl.sqrt(sum_sq)
    
    # Avoid division by zero
    norm_val = tl.where(norm_val > 0.0, norm_val, 1.0)
    
    # Normalize and store
    out = (x_float / norm_val).to(x.dtype)
    tl.store(OUT_ptr + row_start + offsets, out, mask=mask)

@torch.fx.wrap
def triton_l2_norm(in_1):
    nrows = in_1.shape[0]
    ncols = in_1.shape[1]
    
    BLOCK_SIZE = triton.next_power_of_2(ncols)
    
    out = torch.empty_like(in_1)
    
    l2_norm_kernel[(nrows,)](
        X_ptr=in_1,
        OUT_ptr=out,
        nrows=nrows,
        ncols=ncols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_l2_norm