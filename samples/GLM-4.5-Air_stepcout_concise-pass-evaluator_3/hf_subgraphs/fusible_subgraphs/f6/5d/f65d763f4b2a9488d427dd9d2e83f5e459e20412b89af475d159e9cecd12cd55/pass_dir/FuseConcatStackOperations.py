import torch
import triton
import triton.language as tl

@triton.jit
def fused_concat_stack_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    cos_ptr, sin_ptr,
    out_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel for concatenation + stacking operations"""
    # Set up 2D grid for the first dimension (M)
    pid_m = tl.program_id(0)
    
    # Calculate memory offsets for this block
    a_base = a_ptr + pid_m * BLOCK_SIZE_M * N
    b_base = b_ptr + pid_m * BLOCK_SIZE_M * N
    c_base = c_ptr + pid_m * BLOCK_SIZE_M * N
    d_base = d_ptr + pid_m * BLOCK_SIZE_M * N
    
    cos_base = cos_ptr + pid_m * BLOCK_SIZE_M * N
    sin_base = sin_ptr + pid_m * BLOCK_SIZE_M * N
    
    out_base = out_ptr + pid_m * BLOCK_SIZE_M * N * 4  # Final output has 4x more elements
    
    # Create offsets within block
    offsets = tl.arange(0, BLOCK_SIZE_M * N)
    
    # Load first concatenation: cat(a, b) along dim=-1
    a_vals = tl.load(a_base + offsets, mask=offsets < BLOCK_SIZE_M * N, other=0.0)
    b_vals = tl.load(b_base + offsets, mask=offsets < BLOCK_SIZE_M * N, other=0.0)
    concat1 = tl.cat([a_vals, b_vals], axis=0)  # [BLOCK_SIZE_M * 2 * N]
    
    # Load second inputs (cos, sin results)
    cos_vals = tl.load(cos_base + offsets, mask=offsets < BLOCK_SIZE_M * N, other=0.0)
    sin_vals = tl.load(sin_base + offsets, mask=offsets < BLOCK_SIZE_M * N, other=0.0)
    
    # Compute second concatenation: cat(cos, sin) along dim=-1
    concat2 = tl.cat([cos_vals, sin_vals], axis=0)  # [BLOCK_SIZE_M * 2 * N]
    
    # Stack results: create final output where we interleave the concatenated results
    # Output shape: [BLOCK_SIZE_M, 2*N, 2] -> flattened to [BLOCK_SIZE_M * 2 * N * 2]
    out_offsets = tl.arange(0, BLOCK_SIZE_M * 2 * N * 2)
    
    # Create interleaving pattern for stacking operation
    # For each position in the concatenated results, we copy to two positions in output
    pattern_idx = out_offsets % (BLOCK_SIZE_M * 2 * N * 2) // (BLOCK_SIZE_M * 2 * N)
    concat_idx = (out_offsets // (BLOCK_SIZE_M * 2 * N)) % 2
    elem_idx = out_offsets % (BLOCK_SIZE_M * 2 * N)
    
    # Select from appropriate concatenated result
    result = tl.zeros([BLOCK_SIZE_M * 2 * N * 2], dtype=tl.float32)
    
    # From concat1 (stack position 0)
    mask1 = (concat_idx == 0) & (elem_idx < BLOCK_SIZE_M * 2 * N)
    select1 = elem_idx
    values1 = tl.load(concat1 + select1, mask=select1 < BLOCK_SIZE_M * 2 * N, other=0.0)
    result = tl.where(mask1, values1[elem_idx], result)
    
    # From concat2 (stack position 1)  
    mask2 = (concat_idx == 1) & (elem_idx < BLOCK_SIZE_M * 2 * N)
    select2 = elem_idx + BLOCK_SIZE_M * 2 * N  # Offset for second concat result
    values2 = tl.load(concat2 + select2 % (BLOCK_SIZE_M * 2 * N), mask=select2 % (BLOCK_SIZE_M * 2 * N) < BLOCK_SIZE_M * 2 * N, other=0.0)
    result = tl.where(mask2, values2[elem_idx], result)
    
    tl.store(out_base + out_offsets, result, mask=out_offsets < BLOCK_SIZE_M * 2 * N * 2)

@torch.fx.wrap
def fused_concat_stack(a, c, cos, sin):
    """Wrapper for fused concatenation + stacking operations"""
    M, N = a.shape
    BLOCK_SIZE_M = 64  # Optimized for 64x64 tensors
    BLOCK_SIZE_N = N  # Use full N dimension
    
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Output shape should be [M, N*2, 2] 
    out_shape = (M, N * 2 * 2)  # Flattened for storage
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    
    fused_concat_stack_kernel[(num_blocks_m,)](
        a_ptr=a,
        b_ptr=c,     # in_2 concatenated with in_0 
        c_ptr=cos,   # cos result
        d_ptr=sin,   # sin result
        cos_ptr=cos,
        sin_ptr=sin,
        out_ptr=out,
        M=M, N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Reshape to expected [M, N*2, 2] shape
    return out.view(M, N * 2, 2)

def pattern(a, c, cos, sin):
    """Pattern to match concatenation + stacking operations"""
    tmp0 = torch.cat((a, c), dim=-1)
    tmp3 = torch.cat((cos, sin), dim=-1)
    tmp4 = torch.stack((tmp0, tmp3), dim=-1)
    return tmp0, tmp3, tmp4

def replacement_args(a, c, cos, sin):
    """Extract arguments for the fused operation"""
    return (a, c, cos, sin)

def replacement_func():
    """Return the fused concatenation + stacking function"""
    return fused_concat_stack