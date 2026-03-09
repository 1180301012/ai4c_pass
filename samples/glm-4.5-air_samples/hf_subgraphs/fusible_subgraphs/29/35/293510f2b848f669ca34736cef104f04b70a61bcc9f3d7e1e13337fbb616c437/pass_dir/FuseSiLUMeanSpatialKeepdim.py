import torch
import triton
import triton.language as tl

@torch.fx.wrap
def fused_silu_mean_keepdim(x):
    N, C, H, W = x.shape
    spatial_size = H * W
    
    # output shapes
    out_silu = torch.empty_like(x, device=x.device)
    out_mean = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    
    # Optimize block sizes based on tensor shapes
    if C <= 1024:
        BLOCK_SIZE_M = 128
    else:
        BLOCK_SIZE_M = 256
        
    # calculate grid size
    grid_size = (triton.cdiv(C, BLOCK_SIZE_M),)
    
    fused_silu_mean_kernel_keepdim[grid_size](
        x_ptr=x,
        out_silu_ptr=out_silu,
        out_mean_ptr=out_mean,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=1
    )
    
    return out_silu, out_mean

@triton.jit
def fused_silu_mean_kernel_keepdim(
    x_ptr,
    out_silu_ptr,
    out_mean_ptr,
    N,  # batch size
    C,  # channels
    H,  # height  
    W,  # width
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each CTA handles a block of channels
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE_M
    offsets = block_start + tl.arange(0, BLOCK_SIZE_M)
    mask = offsets < C
    
    # Compute spatial mean for this block of channels
    # Initialize with zeros for accumulation
    mean_vals = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Iterate over spatial dimensions
    for h in range(0, H):
        for w in range(0, W):
            # Load input data for this spatial position
            x_ptr_hw = x_ptr + h * W * N * C + w * N * C
            x_block = tl.load(x_ptr_hw + offsets * N, mask=mask, other=0.0)
            
            # Apply SiLU and accumulate
            silu_block = x_block * (1.0 + tl.exp(-x_block))
            mean_vals += silu_block
    
    # Compute final mean by dividing by spatial size
    inv_spatial_size = 1.0 / (H * W)
    mean_vals = mean_vals * inv_spatial_size
    
    # Store mean results (keepdim=True, so 4D tensor)
    tl.store(out_mean_ptr + offsets * H * W * 1, mean_vals, mask=mask)
    
    # Store SiLU output
    for h in range(H):
        for w in range(W):
            out_silu_ptr_hw = out_silu_ptr + h * W * N * C + w * N * C
            silu_block = tl.load(x_ptr_hw + offsets * N, mask=mask, other=0.0)
            silu_out = silu_block * (1.0 + tl.exp(-silu_block))
            tl.store(out_silu_ptr_hw + offsets * N, silu_out, mask=mask)

def pattern(x):
    tmp_0 = torch.nn.functional.silu(x, inplace=True)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)

def replacement_args(x):
    return (x,)

def replacement_func():
    return fused_silu_mean_keepdim