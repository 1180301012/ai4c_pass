import torch
import triton
import triton.language as tl

@triton.jit
def weighted_sum_kernel(
    # Input pointers
    softmax_ptr, in_0_ptr, in_1_ptr,
    # Output pointers
    tmp_3_ptr, tmp_10_ptr,
    # Dimensions
    BATCH: tl.constexpr, CHANNELS: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    # Strides for softmax and tmp_3
    stride_sm_b, stride_sm_c, stride_sm_h, stride_sm_w,
    stride_t3_b, stride_t3_c, stride_t3_h, stride_t3_w,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that:
    1. Stores tmp_3 (reshaped softmax)
    2. Computes weighted sum with in_0 along H dimension
    3. Computes weighted sum with in_1 along W dimension
    4. Stores concatenated result tmp_10
    """
    pid = tl.program_id(0)
    b = pid // CHANNELS
    c = pid % CHANNELS
    
    # Compute output offset for (b, c)
    out_offset = b * CHANNELS + c
    
    # Initialize weighted sums
    sum_0 = 0.0
    sum_1 = 0.0
    
    # Base offsets
    sm_base = b * stride_sm_b + c * stride_sm_c
    t3_base = b * stride_t3_b + c * stride_t3_c
    
    # Iterate over H and W dimensions to compute weighted sums
    # in_0 has shape [1, 1, 1, 64] -> weight along H
    # in_1 has shape [1, 1, 64, 1] -> weight along W
    for h in range(H):
        # Load in_0 weight - shape [1, 1, 1, 64], broadcast to (h, :)
        in_0_weight = tl.load(in_0_ptr + h * 64)  # Access along the 64-dim
        
        for w in range(W):
            # Calculate softmax offset
            sm_offset = sm_base + h * stride_sm_h + w * stride_sm_w
            sm_val = tl.load(softmax_ptr + sm_offset)
            
            # Store to tmp_3 (reshaped view)
            t3_offset = t3_base + h * stride_t3_h + w * stride_t3_w
            tl.store(tmp_3_ptr + t3_offset, sm_val)
            
            # Accumulate weighted sums
            sum_0 += sm_val * in_0_weight
            
            # Load in_1 weight - shape [1, 1, 64, 1], access along the 64-dim
            in_1_weight = tl.load(in_1_ptr + w)
            sum_1 += sm_val * in_1_weight
    
    # Compute output offsets for tmp_10
    tmp_10_offset_0 = out_offset * 2  # First half (sum with in_0)
    tmp_10_offset_1 = out_offset * 2 + 1  # Second half (sum with in_1)
    
    # Store sum results (squeezed dimensions, keepdim=True equivalent)
    # tmp_6 and tmp_9 have shape [batch, 17, 1, 1], stored as scalars at [batch, 17]
    tl.store(tmp_10_ptr + tmp_10_offset_0, sum_0)
    tl.store(tmp_10_ptr + tmp_10_offset_1, sum_1)


def weighted_sum_kernel_wrapper(softmax, in_0, in_1, BATCH, CHANNELS, H, W):
    """Wrapper to launch the fused weighted sum kernel"""
    # Allocate outputs
    # tmp_3 has shape [BATCH, CHANNELS, H, W]
    # tmp_10 has shape [BATCH, CHANNELS, 2] (concatenated sum results)
    tmp_3 = torch.empty((BATCH, CHANNELS, H, W), dtype=softmax.dtype, device=softmax.device)
    tmp_10 = torch.empty((BATCH, CHANNELS, 2), dtype=softmax.dtype, device=softmax.device)
    
    # Get strides
    stride_sm_b, stride_sm_c, stride_sm_h, stride_sm_w = softmax.stride()
    stride_t3_b = BATCH * CHANNELS * H * W  # Contiguous stride
    stride_t3_c = H * W
    stride_t3_h = W
    stride_t3_w = 1
    
    # Calculate grid
    grid = (BATCH * CHANNELS,)
    BLOCK_SIZE = 1024
    
    # Launch kernel
    weighted_sum_kernel[grid](
        softmax, in_0, in_1,
        tmp_3, tmp_10,
        BATCH, CHANNELS, H, W,
        stride_sm_b, stride_sm_c, stride_sm_h, stride_sm_w,
        stride_t3_b, stride_t3_c, stride_t3_h, stride_t3_w,
        BLOCK_SIZE,
    )
    
    return tmp_3, tmp_10


def pattern(softmax, in_0, in_1):
    """
    Match the pattern:
    1. softmax reshape to [BATCH, CHANNELS, H, W]
    2. Two parallel branches: mul + reshape + sum
    3. Concatenate results
    
    The pattern must return both tmp_3 and tmp_10 as they appear in the output.
    """
    tmp_3 = softmax.reshape(-1, 17, 64, 64)
    
    # Branch A with in_0
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(-1, 17, 4096)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    
    # Branch B with in_1
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(-1, 17, 4096)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    
    # Concatenate
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    
    return tmp_3, tmp_10


def replacement_args(softmax, in_0, in_1):
    """Extract dimensions needed for the kernel launch"""
    BATCH = softmax.shape[0]
    CHANNELS = softmax.shape[1]
    H = 64
    W = 64
    return (softmax, in_0, in_1, BATCH, CHANNELS, H, W)


def replacement_func():
    return weighted_sum_kernel_wrapper