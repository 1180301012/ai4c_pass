import torch
import triton
import triton.language as tl

@triton.jit
def slice_expand_kernel_64_128(
    input_ptr, output_ptr,
    B, H, S, D, expand_size,
    stride_ib, stride_ih, stride_is, stride_id,
    stride_ob, stride_oh1, stride_oh2, stride_os, stride_od,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused slice (unsqueeze) + expand kernel
    Input: [B, H, S, D]
    Output: [B, H, expand_size, S, D] where middle dim is broadcasted
    """
    pid = tl.program_id(0)
    total_elements = B * H * S * D
    
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose flat index to [b, h, s, d]
    d_idx = offsets % D
    remainder = offsets // D
    s_idx = remainder % S
    remainder = remainder // S
    h_idx = remainder % H
    b_idx = remainder // H
    
    # Load from input
    in_ptrs = input_ptr + b_idx * stride_ib + h_idx * stride_ih + s_idx * stride_is + d_idx * stride_id
    data = tl.load(in_ptrs, mask=mask, other=0.0)
    
    # Store to output for all expanded positions
    # Output shape: [B, H, expand_size, S, D]
    # We replicate across the expand dimension
    for e in range(expand_size):
        out_ptrs = output_ptr + b_idx * stride_ob + h_idx * stride_oh1 + e * stride_oh2 + s_idx * stride_os + d_idx * stride_od
        tl.store(out_ptrs, data, mask=mask)

@torch.fx.wrap
def optimized_slice_expand_64_128(x):
    """Optimized slice + expand using Triton"""
    B, H, S, D = x.shape
    expand_size = 4
    
    # Output shape: [B, H, expand_size, S, D]
    output = torch.empty(B, H, expand_size, S, D, device=x.device, dtype=x.dtype)
    
    total_elements = B * H * S * D
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    slice_expand_kernel_64_128[grid](
        x, output,
        B, H, S, D, expand_size,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        BLOCK_SIZE,
    )
    
    return output

def pattern(in_2):
    tmp_4 = in_2[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_5 = tmp_4.expand(64, 4, 4, 128, 128)
    return tmp_5

def replacement_args(in_2):
    return (in_2,)

def replacement_func():
    return optimized_slice_expand_64_128