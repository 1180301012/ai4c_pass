import torch
import triton
import triton.language as tl

# Pattern matching
def pattern(in_0, tmp_4):
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    return tmp_7

# Argument extraction
def replacement_args(in_0, tmp_5, tmp_6, tmp_4, tmp_7):
    return (in_0, tmp_4)

# Triton kernel for broadcast multiplication
@triton.jit
def broadcast_mul_kernel(
    in_0_ptr,
    tmp_4_ptr,
    out_ptr,
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread index
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Convert to (b, c, h, w)
    b = idx // (C * H * W)
    c = (idx // (H * W)) % C
    h = (idx // W) % H
    w = idx % W
    
    # Load channel scale
    scale = tl.load(in_0_ptr + c)
    
    # Load value from tmp_4
    val = tl.load(tmp_4_ptr + b*stride_b + c*stride_c + h*stride_h + w*stride_w)
    
    # Multiply
    out_val = scale * val
    
    # Store
    tl.store(out_ptr + b*stride_b + c*stride_c + h*stride_h + w*stride_w, out_val)

# Wrapper function
@torch.fx.wrap
def broadcast_mul(in_0, tmp_4):
    B, C, H, W = tmp_4.shape
    out = torch.empty_like(tmp_4)
    
    # Compute strides
    stride_b, stride_c, stride_h, stride_w = tmp_4.stride()
    
    # Kernel configuration
    BLOCK_SIZE = 128
    num_blocks = (B * C * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    broadcast_mul_kernel[(num_blocks,)](
        in_0,
        tmp_4,
        out,
        B, C, H, W,
        stride_b, stride_c, stride_h, stride_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return broadcast_mul