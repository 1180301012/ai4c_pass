import torch
import triton
import triton.language as tl

# Pattern matching function - match transpose + contiguous + reshape + contiguous
def pattern(x):
    tmp_5 = x.transpose(1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.reshape(1, 257, -1)
    tmp_8 = tmp_7.contiguous()
    return tmp_8

# Argument extraction
def replacement_args(x):
    return (x,)

# Triton kernel for transpose and reshape
# Input: [B, H, Q, D] -> Output: [B, Q, H*D]
@triton.jit
def transpose_reshape_kernel(
    in_ptr,
    out_ptr,
    B, H, Q, D,
    stride_ib, stride_ih, stride_iq, stride_id,
    stride_ob, stride_oq, stride_od,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Total elements = B * Q * H * D
    # Each program handles BLOCK_SIZE elements
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total_elements = B * Q * H * D
    mask = offsets < total_elements
    
    # Convert linear offset to output coordinates [b, q, hd]
    # Output shape: [B, Q, H*D]
    hd = offsets % (H * D)
    tmp = offsets // (H * D)
    q = tmp % Q
    b = tmp // Q
    
    # Split hd into h and d
    h = hd // D
    d = hd % D
    
    # Input offset: [b, h, q, d]
    in_offset = b * stride_ib + h * stride_ih + q * stride_iq + d * stride_id
    
    # Load from input
    vals = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    
    # Output offset: [b, q, h*d]
    out_offset = b * stride_ob + q * stride_oq + hd * stride_od
    
    # Store to output
    tl.store(out_ptr + out_offset, vals, mask=mask)


@torch.fx.wrap
def transpose_reshape(x):
    """
    Fused transpose and reshape:
    Input: [B, H, Q, D] 
    Output: [B, Q, H*D]
    """
    B, H, Q, D = x.shape
    
    # Output shape: [B, Q, H*D]
    output = torch.empty((B, Q, H * D), device=x.device, dtype=x.dtype)
    
    total_elements = B * Q * H * D
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    transpose_reshape_kernel[(num_programs,)](
        x,
        output,
        B, H, Q, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return transpose_reshape