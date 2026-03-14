import torch
import triton
import triton.language as tl


def pattern(in_4):
    """
    Pattern: flatten → transpose
    Matches: tmp_7 = in_4.flatten(2)
             tmp_8 = tmp_7.transpose(1, 2)
    """
    tmp_7 = in_4.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    return tmp_8


def replacement_args(in_4):
    return (in_4,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}, num_warps=8),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def fused_flatten_transpose_kernel(
    in_ptr, out_ptr,
    B, C, HW,
    stride_in_b, stride_in_c, stride_in_hw,
    stride_out_b, stride_out_hw, stride_out_c,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel for flatten + transpose
    Input: [B, C, H, W] → flatten(2) → [B, C, H*W] → transpose(1, 2) → [B, H*W, C]
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Calculate indices
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_hw = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Mask for valid indices
    mask_b = offs_b < B
    mask_hw = offs_hw < HW
    mask_c = offs_c < C
    
    # Create 3D mask
    mask = mask_b[:, None, None] & mask_hw[None, :, None] & mask_c[None, None, :]
    
    # Input: [B, C, HW] (already flattened conceptually)
    # Calculate input pointers
    in_ptrs = (in_ptr + 
               offs_b[:, None, None] * stride_in_b + 
               offs_c[None, None, :] * stride_in_c + 
               offs_hw[None, :, None] * stride_in_hw)
    
    # Load data
    data = tl.load(in_ptrs, mask=mask, other=0.0)
    
    # Output: [B, HW, C] (transposed)
    # Calculate output pointers
    out_ptrs = (out_ptr + 
                offs_b[:, None, None] * stride_out_b + 
                offs_hw[None, :, None] * stride_out_hw + 
                offs_c[None, None, :] * stride_out_c)
    
    # Store data
    tl.store(out_ptrs, data, mask=mask)


@torch.fx.wrap
def fused_flatten_transpose(in_4):
    """
    Fused implementation of flatten + transpose
    in_4: input [B, C, H, W]
    Output: [B, H*W, C]
    """
    B, C, H, W = in_4.shape
    HW = H * W
    
    # Prepare output
    out = torch.empty((B, HW, C), dtype=in_4.dtype, device=in_4.device)
    
    # Grid configuration
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 64
    grid = (triton.cdiv(B, BLOCK_SIZE_B), 
            triton.cdiv(HW, BLOCK_SIZE_M), 
            triton.cdiv(C, BLOCK_SIZE_N))
    
    # Launch kernel
    fused_flatten_transpose_kernel[grid](
        in_4, out,
        B, C, HW,
        in_4.stride(0), in_4.stride(1), 1,  # stride for flatten: [B, C, HW]
        out.stride(0), out.stride(1), out.stride(2),
    )
    
    return out


def replacement_func():
    return fused_flatten_transpose