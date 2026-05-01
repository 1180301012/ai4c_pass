import torch
import triton
import triton.language as tl

def pattern(in_0):
    gelu_result = torch.nn.functional.gelu(in_0)
    mean_result = gelu_result.mean((2, 3), keepdim=True)
    return gelu_result, mean_result

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def gelu_approx(x):
    # Approximate erf(x / sqrt(2)) for gelu
    sqrt_2 = 1.41421356237
    c = 0.5 * x * (1.0 + tl.erf(x / sqrt_2))
    return c

def _compute_gelu_mean(input_ptr, gelu_ptr, mean_ptr, batch, channels, H, W, BLOCK_SIZE: tl.constexpr):
    # Each program handles one spatial element
    pid = tl.program_id(0)
    # Compute 2D spatial index (h, w)
    h = pid // W
    w = pid % W
    
    # Calculate global indices (batch, channel)
    # This kernel runs per (batch, channel) group
    # The grid is (batch * channels * H * W), so:
    batch_idx = pid // (H * W * channels)
    channel_idx = (pid // (H * W)) % channels
    
    # Load input value
    input_val = tl.load(input_ptr + batch_idx * channels * H * W + channel_idx * H * W + h * W + w)
    
    # Compute gelu
    gelu_val = gelu_approx(input_val)
    
    # Store gelu result
    tl.store(gelu_ptr + batch_idx * channels * H * W + channel_idx * H * W + h * W + w, gelu_val)
    
    # Accumulate for mean (using atomic add to avoid global memory reads)
    mean_idx = batch_idx * channels + channel_idx
    tl.atomic_add(mean_ptr + mean_idx, gelu_val)

@torch.fx.wrap
def fused_gelu_mean(input_tensor):
    # Dimensions
    batch, channels, H, W = input_tensor.shape
    
    # Allocate outputs
    gelu_output = torch.empty_like(input_tensor)
    mean_output = torch.empty((batch, channels, 1, 1), dtype=input_tensor.dtype)
    
    # Initialize mean_output to 0
    tl.zeros((batch * channels, ), dtype=input_tensor.dtype, device='cuda').zero_()
    
    # Total elements for kernel launch
    num_elements = batch * channels * H * W
    
    # Launch kernel
    grid = (num_elements,)
    _compute_gelu_mean[(num_elements,)](
        input_tensor, 
        gelu_output, 
        mean_output, 
        batch, channels, H, W, 
        BLOCK_SIZE=1024
    )
    
    # Divide by H*W to get mean
    mean_output = mean_output / (H * W)
    
    return gelu_output, mean_output

def replacement_func():
    return fused_gelu_mean