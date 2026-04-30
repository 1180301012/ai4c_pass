import torch
import triton
import triton.language as tl


# ============ Broadcast Multiply Kernel (fixed configs, no autotuning) ============

@triton.jit
def broadcast_mul_kernel(
    in1_ptr, in2_ptr, out_ptr,
    N, C,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)
    
    mask_n = offs_n < N
    mask_c = offs_c < C
    mask_2d = mask_n[:, None] & mask_c[None, :]
    
    # Load in1 (1D) - contiguous load, broadcast across columns
    in1_vals = tl.load(in1_ptr + offs_n, mask=mask_n, other=0.0)
    
    # Load in2 (2D tile)
    in2_ptrs = in2_ptr + offs_n[:, None] * C + offs_c[None, :]
    in2_vals = tl.load(in2_ptrs, mask=mask_2d, other=0.0)
    
    # Compute out = in1 * in2 (broadcast multiply)
    out_vals = in1_vals[:, None] * in2_vals
    
    # Store output (2D tile)
    out_ptrs = out_ptr + offs_n[:, None] * C + offs_c[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_2d)


@torch.fx.wrap
def _triton_broadcast_mul(in_1, in_2):
    N = in_1.shape[0]
    C = in_2.shape[1]
    
    out = torch.empty_like(in_2)
    
    # Select config based on column size
    if C <= 16:
        BLOCK_N = 128
        BLOCK_C = 16
    elif C <= 32:
        BLOCK_N = 8
        BLOCK_C = 32
    elif C <= 64:
        BLOCK_N = 4
        BLOCK_C = 64
    else:
        BLOCK_N = 4
        BLOCK_C = 128
    
    grid = (triton.cdiv(N, BLOCK_N),)
    
    broadcast_mul_kernel[grid](
        in1_ptr=in_1, in2_ptr=in_2, out_ptr=out,
        N=N, C=C,
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
    )
    
    return out