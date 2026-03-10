import torch
import triton
import triton.language as tl

def pattern(in_2, tmp_1, tmp_0):
    # Match the exact pattern from the original computation
    # torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

def replacement_args(in_2, tmp_1, tmp_0):
    return (in_2, tmp_1, tmp_0)

@triton.jit
def conv2d_1x1_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C_in, H, W, C_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program computes one output spatial location for all output channels
    pid = tl.program_id(0)
    
    # Grid: H * W * N (batch size * height * width)
    h = pid % H
    w = (pid // H) % W
    n = pid // (H * W)
    
    # Load bias first (per output channel)
    bias = tl.load(bias_ptr + tl.arange(0, C_out), mask=True)
    
    # Compute the result for this spatial location
    acc = bias  # Start with bias
    
    # Since it's 1x1 convolution, we just sum over input channels
    for c_in in range(0, C_in, BLOCK_SIZE_K):
        block_size = min(BLOCK_SIZE_K, C_in - c_in)
        weight_block = tl.load(weight_ptr + tl.arange(0, C_out * block_size).to(tl.int64), mask=True)
        
        # Reshape weight to [C_out, block_size]
        weight_reshaped = tl.view(weight_block, [C_out, block_size])
        
        # Load input feature at this spatial location
        input_offset = n * C_in * H * W + c_in * H * W + h * W + w
        input_val = tl.load(x_ptr + input_offset, mask=True)
        
        # Add contribution: weight * input for all output channels
        acc += tl.sum(weight_reshaped * input_val, axis=1)
    
    # Store output
    output_offset = n * C_out * H * W + pid
    tl.store(out_ptr + output_offset, acc)

@torch.fx.wrap
def optimized_conv2d_1x1(x, weight, bias):
    N, C_in, H, W = x.shape
    C_out = bias.shape[0]
    
    # Set tile sizes for optimal performance
    BLOCK_SIZE_M = 32  # Output channels per thread group
    BLOCK_SIZE_N = 1   # Spatial locations per thread group  
    BLOCK_SIZE_K = 32  # Input channels per iteration
    
    # Calculate grid size: total number of spatial locations
    grid_size = N * H * W
    
    output = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    conv2d_1x1_kernel[
        (grid_size,)
    ](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        N=N,
        C_in=C_in,
        H=H,
        W=W,
        C_out=C_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return optimized_conv2d_1x1