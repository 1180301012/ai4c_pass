import torch
import triton
import triton.language as tl


@triton.jit
def fused_residual_gate_kernel(
    in_2_ptr,
    in_0_ptr,
    out_ptr,
    # Tensor dimensions
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    # Strides
    in_2_stride_n: tl.constexpr,
    in_2_stride_c: tl.constexpr,
    in_2_stride_h: tl.constexpr,
    in_2_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    # Block configuration
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    relu -> avg_pool2d -> subtract -> unsqueeze(weight) -> multiply -> add
    """
    # Get position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate tensor size
    total_elements = N * C * H * W
    mask = offsets < total_elements
    
    # Calculate indices
    n = offsets // (C * H * W)
    remainder = offsets % (C * H * W)
    c = remainder // (H * W)
    remainder2 = remainder % (H * W)
    h = remainder2 // W
    w = remainder2 % W
    
    # Load input feature (in_2)
    in_2_offset = n * in_2_stride_n + c * in_2_stride_c + h * in_2_stride_h + w * in_2_stride_w
    x = tl.load(in_2_ptr + in_2_offset, mask=mask, other=0.0)
    
    # ReLU: tmp_2 = relu(x)
    x_relu = tl.where(x > 0, x, 0.0)
    
    # Average pooling 3x3 with padding=1
    # For boundary pixels, we need to check bounds
    pool_sum = tl.constexpr(0.0)
    pool_count = 0
    
    # Unroll the 3x3 pooling loop
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            h_pool = h + kh
            w_pool = w + kw
            # Check bounds (0 <= h < H, 0 <= w < W)
            h_valid = (h_pool >= 0) & (h_pool < H)
            w_valid = (w_pool >= 0) & (w_pool < W)
            valid = h_valid & w_valid
            
            if valid:
                pool_offset = n * in_2_stride_n + c * in_2_stride_c + h_pool * in_2_stride_h + w_pool * in_2_stride_w
                pool_val = tl.load(in_2_ptr + pool_offset, mask=mask, other=0.0)
                pool_val_relu = tl.where(pool_val > 0, pool_val, 0.0)
                pool_sum = pool_sum + pool_val_relu
                pool_count = pool_count + 1
    
    # Average: tmp_3 = pool_sum / pool_count
    pooled_val = pool_sum / pool_count.to(tl.float32)
    
    # Subtract: tmp_4 = tmp_3 - tmp_2 = pooled - relu(x)
    diff = pooled_val - x_relu
    
    # Load weight (in_0) - broadcast from [C] to [N, C, H, W]
    # Weight is at index c
    weight = tl.load(in_0_ptr + c)
    
    # Multiply: tmp_7 = weight * diff
    product = weight * diff
    
    # Add: tmp_8 = tmp_2 + tmp_7 = relu(x) + weight * (pooled - relu(x))
    result = x_relu + product
    
    # Store result
    out_offset = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_residual_gate(in_0, in_1, in_2):
    """
    Fused implementation of:
    tmp_2 = relu(in_2)
    tmp_3 = avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    tmp_5 = in_0.unsqueeze(-1)  # [48] -> [48, 1, 1]
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4  # broadcast multiply
    tmp_8 = tmp_2 + tmp_7
    """
    N, C, H, W = in_2.shape
    
    # Allocate output
    out = torch.empty_like(in_2)
    
    # Block size for parallelism
    BLOCK_SIZE = 1024
    num_programs = N * C * H * W
    
    # Calculate strides
    in_2_stride_n = in_2.stride(0)
    in_2_stride_c = in_2.stride(1)
    in_2_stride_h = in_2.stride(2)
    in_2_stride_w = in_2.stride(3)
    out_stride_n = out.stride(0)
    out_stride_c = out.stride(1)
    out_stride_h = out.stride(2)
    out_stride_w = out.stride(3)
    
    # Launch kernel
    grid = (num_programs,)
    
    fused_residual_gate_kernel[grid](
        in_2,
        in_0,
        out,
        N, C, H, W,
        in_2_stride_n, in_2_stride_c, in_2_stride_h, in_2_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        BLOCK_SIZE,
    )
    
    return out


@triton.jit
def unsqueeze_expand_kernel(
    in_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to expand [C] to [C, H, W] by adding two unsqueeze dimensions.
    This mirrors: tmp_9 = in_1.unsqueeze(-1); tmp_10 = tmp_9.unsqueeze(-1)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C * H * W
    
    c = offsets // (H * W)
    h = (offsets % (H * W)) // W
    w = offsets % W
    
    val = tl.load(in_ptr + c)
    
    out_stride_c = H * W
    out_stride_h = W
    out_stride_w = 1
    out_offset = c * out_stride_c + h * out_stride_h + w * out_stride_w
    
    tl.store(out_ptr + out_offset, val, mask=mask)


@torch.fx.wrap  
def unsqueeze_expand_1d_to_3d(in_1, H, W):
    """
    Expand [C] tensor to [C, H, W] using two unsqueeze operations.
    """
    C = in_1.shape[0]
    out = torch.empty((C, H, W), dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 1024
    num_programs = C * H * W
    grid = (num_programs,)
    
    unsqueeze_expand_kernel[grid](
        in_1,
        out,
        C, H, W,
        BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Match the pattern from float32 models:
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    tmp_5 = tmp_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = tmp_2 + tmp_7
    tmp_9 = tmp_1.unsqueeze(-1)
    tmp_10 = tmp_9.unsqueeze(-1)
    return (tmp_8, tmp_10)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    tmp_5 = tmp_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = tmp_2 + tmp_7
    tmp_9 = tmp_1.unsqueeze(-1)
    tmp_10 = tmp_9.unsqueeze(-1)
    return (tmp_8, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_wrapper
    

@torch.fx.wrap
def fused_wrapper(in_0, in_1, in_2):
    """
    Wrapper that fuses the main computation and handles unsqueeze expansion.
    """
    # Fused main computation: relu + pool + subtract + multiply + add
    tmp_8 = fused_residual_gate(in_0, in_1, in_2)
    
    # Handle second unsqueeze expansion (in_1 -> [C, 1, 1])
    N, C, H, W = in_2.shape
    tmp_10 = unsqueeze_expand_1d_to_3d(in_1, 1, 1)
    
    return (tmp_8, tmp_10)