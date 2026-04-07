import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation in model.py
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel for fused GELU and mean computation
@triton.jit
def fused_gelu_mean_kernel(
    in_ptr,
    gelu_out_ptr, 
    mean_out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel that computes GELU and mean reduction simultaneously"""
    # Program ID for each thread
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    
    # Each thread processes a contiguous block
    block_size = BLOCK_SIZE_M * BLOCK_SIZE_N
    block_start = pid * block_size
    block_end = min(block_start + block_size, total_elements)
    
    # Initialize accumulators
    local_sum = 0.0
    local_count = 0
    
    # Process the assigned block
    for idx in range(block_start, block_end):
        # Calculate coordinates from linear index  
        batch_idx = idx // (channels * height * width)
        remainder = idx % (channels * height * width)
        channel_idx = remainder // (height * width)
        remainder = remainder % (height * width)
        h_idx = remainder // width
        w_idx = remainder % width
        
        # Load input element
        in_ptr_elem = in_ptr + batch_idx * channels * height * width + channel_idx * height * width + h_idx * width + w_idx
        x = tl.load(in_ptr_elem)
        
        # Compute GELU using simple sigmoid approximation (avoiding problematic tanh)
        # GELU approximation: x * sigmoid(1.702 * x)
        x_scaled = 1.702 * x
        sigmoid_approx = 1.0 / (1.0 + tl.exp(-x_scaled))
        gelu_val = x * sigmoid_approx
        
        # Store GELU output
        gelu_out_ptr_elem = gelu_out_ptr + idx
        tl.store(gelu_out_ptr_elem, gelu_val)
        
        # Accumulate for mean calculation
        local_sum += gelu_val
        local_count += 1
    
    # Store mean value for the corresponding batch and channel
    if local_count > 0:
        mean_val = local_sum / local_count
        # Calculate mean output index (batch, channel, 1, 1 -> flattened)
        mean_idx = pid // (height * width)  # This gives us (batch * channels) index
        tl.store(mean_out_ptr + mean_idx, mean_val)

@torch.fx.wrap
def fused_gelu_mean_forward(in_tensor):
    """Wrapper function to launch the fused kernel"""
    
    # Get input tensor properties
    batch_size, channels, height, width = in_tensor.shape
    total_elements = batch_size * channels * height * width
    
    # Create output tensors
    gelu_out = torch.empty_like(in_tensor)
    mean_out = torch.empty(batch_size, channels, 1, 1, dtype=in_tensor.dtype, device=in_tensor.device)
    mean_out_flat = mean_out.view(batch_size * channels)
    
    # Triton kernel launch configuration
    BLOCK_SIZE_M = 16  # Block size for spatial dimensions
    BLOCK_SIZE_N = 16
    block_size = BLOCK_SIZE_M * BLOCK_SIZE_N
    num_programs = (total_elements + block_size - 1) // block_size
    
    # Launch Triton kernel
    fused_gelu_mean_kernel[(num_programs,)](
        in_ptr=in_tensor,
        gelu_out_ptr=gelu_out,
        mean_out_ptr=mean_out_flat,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return gelu_out, mean_out

# Replacement function (returns function reference, not the actual call)
def replacement_func():
    return fused_gelu_mean_forward