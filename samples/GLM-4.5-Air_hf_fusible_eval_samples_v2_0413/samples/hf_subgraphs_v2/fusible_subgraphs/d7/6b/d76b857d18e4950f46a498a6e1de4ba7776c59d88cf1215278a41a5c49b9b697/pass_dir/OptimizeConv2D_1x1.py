import torch
import triton
import triton.language as tl

# Pattern matching function for 1x1 conv2d
def conv2d_pattern(in_0, in_1, in_2):
    """
    Pattern: 1x1 convolution with bias
    Args:
        in_0: bias tensor [num_output]
        in_1: weight tensor [num_output, in_channels, 1, 1]
        in_2: input tensor [batch_size, in_channels, height, width]
    """
    result = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return result

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """Extract arguments for optimized 1x1 conv2d"""
    return (in_0, in_1, in_2)

# Optimized 1x1 convolution kernel
@triton.jit
def conv1x1_kernel(
    # Pointers to input tensors
    x_ptr,          # [N, C, H, W] input
    weight_ptr,     # [O, C, 1, 1] weights
    bias_ptr,       # [O] bias
    output_ptr,     # [N, O, H, W] output
    
    # Tensor shapes
    N,
    C,
    H, W,
    O,              # Output channels
    
    # Stride and padding info
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program id
    pid = tl.program_id(0)
    
    # Grid boundaries
    grid_m = (N * O + BLOCK_M - 1) // BLOCK_M
    grid_n = (H * W + BLOCK_N - 1) // BLOCK_N
    pid_m = pid % grid_m
    pid_n = pid // grid_m
    
    # Get batch and channel index
    n = pid_m // (O // groups)
    o_base = (pid_m % (O // groups)) * groups + (n % groups)
    o = o_base
    
    # Get spatial indices
    h_idx = pid_n // (W // BLOCK_N)
    w_idx = pid_n % (W // BLOCK_N)
    
    # Load bias
    bias = tl.load(bias_ptr + o, other=0.0)
    
    # Initialize accumulator
    acc = bias
    
    # Vectorization factor
    vector_size = 4
    
    # Process input channels
    for k in range(0, C, BLOCK_K):
        k_end = min(k + BLOCK_K, C)
        
        # Load weights (1x1 conv, so no spatial)
        weight = tl.load(weight_ptr + o * C + k, other=0.0)
        
        # Load input values
        h_off = h_idx * BLOCK_N
        w_off = w_idx * BLOCK_N
        
        for i in range(BLOCK_N):
            for j in range(0, BLOCK_N, vector_size):
                # Current spatial position
                curr_h = h_off + i
                curr_w = w_off + j
                
                # Mask to stay within bounds
                if curr_h < H and curr_w < W:
                    # Load input tensor values with vectorization
                    input_ptrs = x_ptr + n * C * H * W + k * H * W + curr_h * W + curr_w + tl.arange(0, vector_size)
                    input_vals = tl.load(input_ptrs, mask=tl.arange(0, vector_size) < min(BLOCK_N - j, vector_size), other=0.0)
                    
                    # Multiply and accumulate
                    acc += input_vals * weight
    
    # Store result
    output_ptrs = output_ptr + n * O * H * W + o * H * W + h_idx * BLOCK_N * W + w_idx * BLOCK_N
    tl.store(output_ptrs, acc, mask=tl.arange(0, BLOCK_N) < min(BLOCK_N, W - w_idx * BLOCK_N))

@torch.fx.wrap
def optimized_conv1x1(in_0, in_1, in_2):
    """Optimized 1x1 convolution wrapper"""
    # Get tensor shapes
    N, C, H, W = in_2.shape
    O, C, K_H, K_W = in_1.shape
    
    # Create output tensor
    output = torch.empty((N, O, H, W), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel with optimized block sizes
    BLOCK_M = 128  # Process output channels efficiently
    BLOCK_N = 64   # Process spatial tiles efficiently
    BLOCK_K = 32   # Process input channels efficiently
    
    # Calculate grid size
    grid_m = (N * O + BLOCK_M - 1) // BLOCK_M
    grid_n = (H * W + BLOCK_N - 1) // BLOCK_N
    total_programs = grid_m * grid_n
    
    # Only launch if we have valid dimensions
    if total_programs > 0:
        conv1x1_kernel[(total_programs,)](
            in_2,      # input_ptr
            in_1,      # weight_ptr
            in_0,      # bias_ptr
            output,    # output_ptr
            N, C, H, W,
            O,
            1, 1,  stride_h, stride_w,  # dilation_h, dilation_w, stride_h, stride_w
            0, 0,     # pad_h, pad_w
            1,        # groups
            BLOCK_M, BLOCK_N, BLOCK_K,
        )
    
    return output

# Replacement function
def replacement_func():
    """Return optimized 1x1 convolution function"""
    return optimized_conv1x1