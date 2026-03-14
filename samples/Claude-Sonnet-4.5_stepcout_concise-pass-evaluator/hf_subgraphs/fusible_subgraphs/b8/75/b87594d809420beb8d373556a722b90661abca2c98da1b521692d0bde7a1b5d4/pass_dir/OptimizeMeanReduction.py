import torch
import triton
import triton.language as tl

# Pattern matching function for mean reduction along dimension -2
def pattern(x):
    """
    Match mean reduction along dimension -2
    Input: [B, N, C] tensor
    Output: [B, C] tensor
    """
    result = x.mean(-2)
    return result

def replacement_args(x):
    return (x,)

# Ultra-simplified kernel
@triton.jit
def ultra_fast_mean_kernel(
    x_ptr, out_ptr,
    B, N, C,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute batch and channel block IDs
    num_blocks = tl.cdiv(C, BLOCK_C)
    bid = pid // num_blocks
    cid = pid % num_blocks
    
    # Channel indices
    c_offs = cid * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = c_offs < C
    
    # Accumulate
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    base = bid * N * C
    
    for n in range(N):
        ptr = base + n * C + c_offs
        val = tl.load(x_ptr + ptr, mask=mask, other=0.0)
        acc += val
    
    # Mean and store
    out = acc * (1.0 / N)
    out_ptr_idx = bid * C + c_offs
    tl.store(out_ptr + out_ptr_idx, out, mask=mask)

@torch.fx.wrap
def optimized_mean(x):
    """
    Ultra-fast mean reduction
    """
    x = x.contiguous()
    B, N, C = x.shape
    out = torch.empty((B, C), device=x.device, dtype=x.dtype)
    
    # Use 256 for better vectorization
    BLOCK_C = 256
    grid = (B * triton.cdiv(C, BLOCK_C),)
    
    ultra_fast_mean_kernel[grid](x, out, B, N, C, BLOCK_C=BLOCK_C)
    
    return out

def replacement_func():
    return optimized_mean