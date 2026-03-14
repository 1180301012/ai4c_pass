import torch
import triton
import triton.language as tl

# Pattern matching function - matches SiLU + mean pattern
def pattern(in_0):
    """
    Match the computation pattern:
    1. torch.nn.functional.silu(in_0, inplace=True)
    2. tmp_0.mean((2, 3), keepdim=True/False)
    
    Returns both the activated tensor and the mean.
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_1


def replacement_args(in_0):
    # Extract arguments needed for replacement
    return (in_0,)


# Optimized Triton kernel that fuses SiLU activation with mean reduction
@triton.jit
def fused_silu_mean_kernel(
    input_ptr,
    mean_output_ptr,
    activated_output_ptr,
    batch_stride, channel_stride, height_stride, width_stride,
    # Output strides
    mean_batch_stride, mean_channel_stride,
    activated_batch_stride, activated_channel_stride, activated_height_stride, activated_width_stride,
    # Dimensions
    N, C, H, W,
    # Mean reduction dimensions (H, W)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes SiLU activation and mean reduction in a single pass.
    """
    # Get position
    pid = tl.program_id(0)
    num_pid_h = tl.cdiv(C, BLOCK_SIZE)
    pid_b = pid // num_pid_h
    pid_c = pid % num_pid_h
    
    # Calculate channel offset
    channel_offset = pid_c * BLOCK_SIZE
    
    # Input pointer for the channel
    input_ptr_offset = (
        pid_b * batch_stride + 
        channel_offset * channel_stride
    )
    
    # Create offsets for height and width
    offs_h = tl.arange(0, BLOCK_SIZE)
    offs_w = tl.arange(0, BLOCK_SIZE)
    
    # Initialize accumulator for mean
    accumulator = 0.0
    count = 0.0
    
    # Mean output pointer
    mean_ptr_offset = pid_b * mean_batch_stride + channel_offset * mean_channel_stride
    mean_ptr = mean_output_ptr + mean_ptr_offset
    
    # Activated output pointer (for this channel)
    activated_ptr_base = activated_output_ptr + pid_b * activated_batch_stride + channel_offset * activated_channel_stride
    
    # Process each height position
    for h in range(H):
        for w in range(W):
            # Check bounds
            if h < H and w < W and channel_offset + tl.arange(0, BLOCK_SIZE) < C:
                # Load input value
                input_offset = input_ptr_offset + h * height_stride + w * width_stride
                
                # Vectorized load
                offs = channel_offset + offs_h * 0  # Just for mask
                
                # Load and compute SiLU for each element in the channel
                for bh in range(BLOCK_SIZE):
                    c_idx = channel_offset + bh
                    if c_idx < C and h < H and w < W:
                        # Load value
                        ptr = input_ptr + pid_b * batch_stride + c_idx * channel_stride + h * height_stride + w * width_stride
                        x = tl.load(ptr)
                        
                        # SiLU activation: x * sigmoid(x)
                        sigmoid = 1.0 / (1.0 + tl.exp(-x))
                        silu_out = x * sigmoid
                        
                        # Accumulate for mean
                        accumulator += silu_out
                        count += 1.0
                        
                        # Store activated output
                        out_ptr = activated_output_ptr + pid_b * activated_batch_stride + c_idx * activated_channel_stride + h * activated_height_stride + w * activated_width_stride
                        tl.store(out_ptr, silu_out)
    
    # Compute mean and store
    mean_val = accumulator / count
    # Store mean (keepdim=True case - shape [N, C, 1, 1])
    for bh in range(BLOCK_SIZE):
        c_idx = channel_offset + bh
        if c_idx < C:
            mean_out_ptr = mean_output_ptr + pid_b * mean_batch_stride + c_idx * mean_channel_stride
            tl.store(mean_out_ptr, mean_val)


# Better optimized version with proper vectorization
@triton.jit
def fused_silu_mean_kernel_v2(
    input_ptr,
    mean_output_ptr,
    activated_output_ptr,
    N, C, H, W,
    mean_bs_stride, mean_c_stride,
    act_bs_stride, act_c_stride, act_h_stride, act_w_stride,
    in_bs_stride, in_c_stride, in_h_stride, in_w_stride,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Optimized fused kernel with 2D blocking for H and W dimensions.
    """
    # Get positions
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Calculate offsets
    off_b = pid_b
    off_c = pid_c
    off_h = pid_h * BLOCK_SIZE_H
    
    # Initialize accumulator
    accum = 0.0
    count = 0.0
    
    # Process height blocks
    for h in range(BLOCK_SIZE_H):
        actual_h = off_h + h
        if actual_h >= H:
            break
            
        for w in range(W):
            # Load input
            input_offset = (off_b * in_bs_stride + off_c * in_c_stride + 
                          actual_h * in_h_stride + w * in_w_stride)
            x = tl.load(input_ptr + input_offset)
            
            # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            sigmoid = 1.0 / (1.0 + tl.exp(-x))
            silu_out = x * sigmoid
            
            # Accumulate for mean
            accum += silu_out
            count += 1.0
            
            # Store activated output
            act_offset = (off_b * act_bs_stride + off_c * act_c_stride + 
                         actual_h * act_h_stride + w * act_w_stride)
            tl.store(activated_output_ptr + act_offset, silu_out)
    
    # Compute mean
    mean_val = accum / count
    
    # Store mean
    mean_offset = off_b * mean_bs_stride + off_c * mean_c_stride
    tl.store(mean_output_ptr + mean_offset, mean_val)


# Version 3: Simple but efficient - process one channel at a time with vectorized loads
@triton.jit
def fused_silu_mean_kernel_v3(
    input_ptr,
    mean_output_ptr,
    activated_output_ptr,
    # Dimensions
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    # Strides
    in_bs_stride: tl.constexpr, in_c_stride: tl.constexpr, 
    in_h_stride: tl.constexpr, in_w_stride: tl.constexpr,
    act_bs_stride: tl.constexpr, act_c_stride: tl.constexpr,
    act_h_stride: tl.constexpr, act_w_stride: tl.constexpr,
    mean_bs_stride: tl.constexpr, mean_c_stride: tl.constexpr,
    # Block size
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """
    Efficient fused kernel for SiLU + mean computation.
    """
    # Get program id
    pid = tl.program_id(0)
    num_channels = C
    
    # Calculate channel
    c = pid % num_channels
    b = pid // num_channels
    
    if b >= N:
        return
    
    # Pointers
    in_ptr = input_ptr + b * in_bs_stride + c * in_c_stride
    act_ptr = activated_output_ptr + b * act_bs_stride + c * act_c_stride
    mean_ptr = mean_output_ptr + b * mean_bs_stride + c * mean_c_stride
    
    # Accumulator
    accum = 0.0
    count = 0.0
    
    # Process each spatial location
    for h in range(H):
        for w in range(W):
            # Load
            x = tl.load(in_ptr + h * in_h_stride + w * in_w_stride)
            
            # SiLU: x / (1 + exp(-x))
            silu = x / (1.0 + tl.exp(-x))
            
            # Accumulate
            accum += silu
            count += 1.0
            
            # Store activated
            tl.store(act_ptr + h * act_h_stride + w * act_w_stride, silu)
    
    # Store mean
    tl.store(mean_ptr, accum / count)


@torch.fx.wrap
def fused_silu_mean_kernel_wrapper(x):
    """
    Wrapper function that launches the fused kernel.
    Returns: (activated_tensor, mean_tensor)
    """
    N, C, H, W = x.shape
    
    # Allocate outputs
    activated = torch.empty_like(x)
    mean = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)
    
    # Calculate strides
    in_bs, in_c, in_h, in_w = x.stride()
    act_bs, act_c, act_h, act_w = activated.stride()
    mean_bs, mean_c = mean.stride()
    
    # Grid: (N * C,) - each program processes one channel of one batch
    grid = (N * C,)
    
    # Launch kernel
    fused_silu_mean_kernel_v3[grid](
        x, mean, activated,
        N, C, H, W,
        in_bs, in_c, in_h, in_w,
        act_bs, act_c, act_h, act_w,
        mean_bs, mean_c,
        BLOCK_H=1, BLOCK_W=1,
    )
    
    return activated, mean


def replacement_func():
    return fused_silu_mean_kernel_wrapper