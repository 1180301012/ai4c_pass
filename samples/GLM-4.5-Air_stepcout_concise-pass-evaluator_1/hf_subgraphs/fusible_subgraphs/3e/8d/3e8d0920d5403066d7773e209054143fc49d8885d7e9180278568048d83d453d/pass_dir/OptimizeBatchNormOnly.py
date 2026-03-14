import torch
import triton
import triton.language as tl


def pattern(input_in, running_mean, running_var, weight, bias):
    tmp_8 = torch.nn.functional.batch_norm(input_in, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    return tmp_8


def replacement_args(input_in, running_mean, running_var, weight, bias):
    return (input_in, running_mean, running_var, weight, bias)


@triton.jit
def optimized_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Calculate grid coordinates
    m = tl.program_id(0)  # height position
    n = tl.program_id(1)  # width position
    k = tl.program_id(2)  # channel block
    
    # Each thread processes a block of channels for one pixel
    batch_base = k * BLOCK_SIZE_K
    batch_end = min(batch_base + BLOCK_SIZE_K, n_channels)
    
    # Calculate position in output
    output_offset = m * width * n_channels + n * n_channels
    
    # Process each channel in the block
    for c in range(batch_base, batch_end):
        load_offset = output_offset + c
        x = tl.load(input_ptr + load_offset)
        
        # Load batch normalization parameters precisely aligned with iteration
        channel_idx = c
        mean = tl.load(running_mean_ptr + channel_idx).to(tl.float32)
        var = tl.load(running_var_ptr + channel_idx).to(tl.float32)
        weight_val = tl.load(weight_ptr + channel_idx).to(tl.float32)
        bias_val = tl.load(bias_ptr + channel_idx).to(tl.float32)
        
        # Use exact PyTorch formula: γ * (x - μ) / sqrt(σ² + ε) + β
        eps = 0.001
        
        # Add epsilon to variance to avoid division by zero
        var_with_eps = var + eps
        
        # Compute sqrt in a numerically stable way
        sqrt_var = tl.sqrt(var_with_eps)
        
        # Perform the normalization: (x - mean) / sqrt_var
        # Use max to ensure sqrt_var never gets too small
        if sqrt_var > 1e-7:
            normalized = (x - mean) / sqrt_var
        else:
            normalized = x - mean
        
        # Apply weight and bias
        result = normalized * weight_val + bias_val
        
        # Store result
        tl.store(output_ptr + load_offset, result)


@torch.fx.wrap
def optimized_batch_norm(input_in, running_mean, running_var, weight, bias):
    # Get tensor shapes
    n_channels, height, width = input_in.shape[0], input_in.shape[2], input_in.shape[3]
    
    # Create output tensor
    output_shape = input_in.shape
    output = torch.empty(output_shape, dtype=input_in.dtype, device=input_in.device)
    
    # Configure block sizes for better GPU utilization
    BLOCK_SIZE_M = 8   # Process 8 rows per block for better occupancy
    BLOCK_SIZE_N = 8   # Process 8 columns per block for better occupancy
    BLOCK_SIZE_K = min(1024, n_channels)  # Process more channels per block
    
    # Calculate grid size
    grid_m = (height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (n_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    grid = (grid_m, grid_n, grid_k)
    
    # Launch kernel
    optimized_batch_norm_kernel[grid](
        input_ptr=input_in,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output


def replacement_func():
    return optimized_batch_norm