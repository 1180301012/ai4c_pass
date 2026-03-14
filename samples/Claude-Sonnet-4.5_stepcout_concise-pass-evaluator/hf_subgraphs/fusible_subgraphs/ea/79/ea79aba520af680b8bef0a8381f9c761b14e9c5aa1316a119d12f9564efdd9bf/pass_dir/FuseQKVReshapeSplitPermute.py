import torch
import triton
import triton.language as tl

def pattern(tmp_7):
    """Match permute followed by transpose"""
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_13 = tmp_10.transpose(-2, -1)
    return tmp_13

def replacement_args(tmp_7):
    return (tmp_7,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE_D': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_D': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_D': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_D': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 64}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def fused_permute_transpose_kernel(
    input_ptr, output_ptr,
    B, S, H, D,
    stride_i_b, stride_i_s, stride_i_h, stride_i_d,
    stride_o_b, stride_o_h, stride_o_d, stride_o_s,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Fuse permute(0, 2, 1, 3) followed by transpose(-2, -1)
    Input: [B, S, H, D]
    Output: [B, H, D, S]
    
    Process one (b, h, s) tuple per program, vectorize over D
    """
    pid = tl.program_id(0)
    
    total_instances = B * H * S
    if pid >= total_instances:
        return
    
    # Decompose pid to (b, h, s)
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    s = rem % S
    
    # Vectorized copy over D dimension
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    mask = d_offsets < D
    
    # Read from input[b, s, h, :]
    input_base = b * stride_i_b + s * stride_i_s + h * stride_i_h
    input_offsets = input_base + d_offsets * stride_i_d
    vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Write to output[b, h, :, s]
    output_base = b * stride_o_b + h * stride_o_h + s * stride_o_s
    output_offsets = output_base + d_offsets * stride_o_d
    tl.store(output_ptr + output_offsets, vals, mask=mask)

@torch.fx.wrap
def fused_permute_transpose(tmp_7):
    """Optimized permute+transpose fusion"""
    B, S, H, D = tmp_7.shape
    
    # Allocate output tensor [B, H, D, S]
    output = torch.empty((B, H, D, S), dtype=tmp_7.dtype, device=tmp_7.device)
    
    # Launch kernel with one program per (b, h, s), vectorize over D
    grid = (B * H * S,)
    
    fused_permute_transpose_kernel[grid](
        tmp_7, output,
        B, S, H, D,
        tmp_7.stride(0), tmp_7.stride(1), tmp_7.stride(2), tmp_7.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return output

def replacement_func():
    return fused_permute_transpose