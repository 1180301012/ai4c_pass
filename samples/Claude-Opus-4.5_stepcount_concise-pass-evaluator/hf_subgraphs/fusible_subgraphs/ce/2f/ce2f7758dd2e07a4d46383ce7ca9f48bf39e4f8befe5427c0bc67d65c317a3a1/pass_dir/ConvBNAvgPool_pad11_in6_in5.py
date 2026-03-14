import torch
import triton
import triton.language as tl

# Pattern: avg_pool2d only
def pattern(x):
    tmp_7 = torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)
    return tmp_7

def replacement_args(x):
    return (x,)

# Simple fixed-config kernel for minimal overhead
@triton.jit
def avg_pool2d_kernel(
    input_ptr, output_ptr,
    NC, W, HW, W_out, HW_out, n_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_out
    
    # Decompose output index
    nc = offs // HW_out
    hw_out = offs % HW_out
    h_out = hw_out // W_out
    w_out = hw_out % W_out
    
    # Input base index
    base = nc * HW + (h_out * 2) * W + (w_out * 2)
    
    # Load 2x2 and average
    v0 = tl.load(input_ptr + base, mask=mask, other=0.0)
    v1 = tl.load(input_ptr + base + 1, mask=mask, other=0.0)
    v2 = tl.load(input_ptr + base + W, mask=mask, other=0.0)
    v3 = tl.load(input_ptr + base + W + 1, mask=mask, other=0.0)
    
    out = (v0 + v1 + v2 + v3) * 0.25
    tl.store(output_ptr + offs, out, mask=mask)

@torch.fx.wrap
def optimized_avg_pool2d(x):
    N, C, H, W = x.shape
    H_out = H // 2
    W_out = W // 2
    
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    NC = N * C
    HW = H * W
    HW_out = H_out * W_out
    n_out = NC * HW_out
    
    # Fixed block size - balance between parallelism and overhead
    BLOCK = 1024
    grid = ((n_out + BLOCK - 1) // BLOCK,)
    
    avg_pool2d_kernel[grid](
        x, out,
        NC, W, HW, W_out, HW_out, n_out,
        BLOCK_SIZE=BLOCK,
    )
    
    return out

def replacement_func():
    return optimized_avg_pool2d