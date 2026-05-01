import torch
import triton
import triton.language as tl

# Pattern matching
def pattern(in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    return tmp_4

# Argument extraction
def replacement_args(in_2, tmp_2, tmp_3, tmp_4):
    return (tmp_2,)

# Triton kernel for fused avg_pool2d and subtraction
@triton.jit
def fused_avg_pool_subtract_kernel(
    in_ptr,
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
    
    # Load current value
    current = tl.load(in_ptr + b*stride_b + c*stride_c + h*stride_h + w*stride_w)
    
    # Compute 3x3 average
    total = 0.0
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            ih = h + i
            iw = w + j
            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                val = tl.load(in_ptr + b*stride_b + c*stride_c + ih*stride_h + iw*stride_w)
                total += val
                count += 1
    avg = total / tl.cast(count, tl.float32)
    
    # Subtract
    out_val = avg - current
    
    # Store
    tl.store(out_ptr + b*stride_b + c*stride_c + h*stride_h + w*stride_w, out_val)

# Wrapper function
@torch.fx.wrap
def fused_avg_pool_subtract(re_lu_out):
    B, C, H, W = re_lu_out.shape
    out = torch.empty_like(re_lu_out)
    
    # Compute strides
    stride_b, stride_c, stride_h, stride_w = re_lu_out.stride()
    
    # Kernel configuration
    BLOCK_SIZE = 128
    num_blocks = (B * C * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_avg_pool_subtract_kernel[(num_blocks,)](
        re_lu_out,
        out,
        B, C, H, W,
        stride_b, stride_c, stride_h, stride_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_avg_pool_subtract