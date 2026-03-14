import torch
import triton
import triton.language as tl

def pattern(conv_bias, conv_weight, input_tensor):
    """
    Pattern matching for attention computation with softmax optimization for large spatial dimensions
    This matches the same pattern but includes specific optimizations for large H*W cases
    """
    # 1x1 convolution with the exact parameters used in the target graphs
    conv_output = torch.conv2d(input_tensor, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Get the shape after conv and compute flattened dimension
    batch_size, channels, height, width = conv_output.shape
    flattened_dim = height * width
    
    # Reshape to (batch, 1, flattened_dim)
    reshaped = conv_output.view(batch_size, 1, flattened_dim)
    
    # Softmax along dimension 2 (over spatial locations)
    softmax_output = torch.nn.functional.softmax(reshaped, 2, _stacklevel=5)
    
    # Add final dimension
    final_output = softmax_output.unsqueeze(-1)
    
    return final_output

def replacement_args(conv_bias, conv_weight, input_tensor):
    """Extract arguments needed for the replacement kernel"""
    batch_size, channels, height, width = input_tensor.shape
    flattened_dim = height * width
    total_elements = batch_size * flattened_dim
    return (conv_bias, conv_weight, input_tensor, batch_size, flattened_dim, total_elements, channels)

@triton.jit
def high_dim_softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    flattened_dim,
    in_channels,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr
):
    """Highly optimized softmax kernel for large spatial dimensions"""
    # Program identifiers
    pid = tl.program_id(0)
    total_programs = batch_size * (flattened_dim + BLOCK_COLS - 1) // BLOCK_COLS
    
    if pid >= total_programs:
        return
    
    # Convert program ID to batch and spatial coordinates
    batch_idx = pid // ((flattened_dim + BLOCK_COLS - 1) // BLOCK_COLS)
    spatial_block_idx = pid % ((flattened_dim + BLOCK_COLS - 1) // BLOCK_COLS)
    
    # Spatial range for this program
    spatial_start = spatial_block_idx * BLOCK_COLS
    spatial_end = min(spatial_start + BLOCK_COLS, flattened_dim)
    
    # Process each batch element
    if batch_idx < batch_size:
        # Load and process for this batch
        row_offset = batch_idx * flattened_dim
        
        # Compute weighted sums for this batch across spatial dimensions
        max_val = -tl.float32('inf')
        sum_val = 0.0
        
        # First pass: compute max and sum
        for spatial_idx in range(spatial_start, spatial_end):
            # Compute conv output for this (batch, spatial) location
            weighted_sum = 0.0
            
            # Channel reduction for 1x1 conv with weight [1, C, 1, 1]
            for c in range(in_channels):
                input_c_idx = row_offset + spatial_idx + c * batch_size * flattened_dim
                weight_c_idx = c
                
                input_val = tl.load(input_ptr + input_c_idx, mask=(input_c_idx < batch_size * in_channels * flattened_dim), other=0.0)
                weight_val = tl.load(input_ptr + weight_c_idx, mask=(weight_c_idx < in_channels), other=0.0)
                weighted_sum += weight_val * input_val
            
            # Load bias
            bias_val = tl.load(input_ptr + batch_idx, mask=(batch_idx < batch_size), other=0.0)
            total = weighted_sum + bias_val
            
            max_val = tl.maximum(max_val, total)
        
        # Second pass: compute softmax with max stability
        for spatial_idx in range(spatial_start, spatial_end):
            # Compute conv output again (could be optimized with shared memory)
            weighted_sum = 0.0
            for c in range(in_channels):
                input_c_idx = row_offset + spatial_idx + c * batch_size * flattened_dim
                weight_c_idx = c
                
                input_val = tl.load(input_ptr + input_c_idx, mask=(input_c_idx < batch_size * in_channels * flattened_dim), other=0.0)
                weight_val = tl.load(input_ptr + weight_c_idx, mask=(weight_c_idx < in_channels), other=0.0)
                weighted_sum += weight_val * input_val
            
            bias_val = tl.load(input_ptr + batch_idx, mask=(batch_idx < batch_size), other=0.0)
            total = weighted_sum + bias_val
            
            # Softmax computation
            exp_val = tl.exp(total - max_val)
            sum_val += exp_val
            store_idx = row_offset + spatial_idx
            tl.store(output_ptr + store_idx, exp_val, mask=(store_idx < batch_size * flattened_dim))
        
        # Normalize (this part could be done in a separate kernel or with atomic ops)
        # For simplicity, we'll do normalization in the softmax wrapper function
        # Divide by sum_val for each element in the row

@triton.jit
def normalize_kernel(
    intermediate_ptr,
    final_output_ptr,
    batch_size,
    flattened_dim,
    norm_ptr,
    BLOCK_SIZE: tl.constexpr
):
    """Normalization kernel for softmax"""
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    row_offset = pid * flattened_dim
    norm_val = tl.load(norm_ptr + pid, mask=(pid < batch_size), other=1.0)
    
    for i in range(flattened_dim):
        if i < BLOCK_SIZE:
            idx = row_offset + i
            if idx < batch_size * flattened_dim:
                intermediate = tl.load(intermediate_ptr + idx, other=0.0)
                tl.store(final_output_ptr + idx, intermediate / norm_val)

def high_dim_attention_optimization(conv_bias, conv_weight, input_tensor, batch_size, flattened_dim, total_elements, in_channels):
    """Optimized function for high-dimensional attention computation"""
    device = input_tensor.device
    
    # Allocate intermediate output tensor
    intermediate = torch.empty(total_elements, dtype=torch.float32, device=device)
    
    # Choose optimal block sizes based on flattened dimension
    if flattened_dim >= 2048:
        BLOCK_COLS = 512
        BLOCK_ROWS = 1
    elif flattened_dim >= 1024:
        BLOCK_COLS = 256
        BLOCK_ROWS = 1
    else:
        BLOCK_COLS = 128
        BLOCK_ROWS = 1
    
    # Calculate grid dimensions
    total_programs = batch_size * (flattened_dim + BLOCK_COLS - 1) // BLOCK_COLS
    
    # Create input buffer combining bias, weight, and input data
    # We need to structure the data efficiently for the kernel
    # For simplicity, we'll use a more direct approach
    
    if total_elements <= 4096:  # Use PyTorch's optimized softmax for small tensors
        # Direct convolution + softmax for small cases
        conv_output = torch.conv2d(input_tensor, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
        reshaped = conv_output.view(batch_size, 1, flattened_dim)
        softmax_output = torch.softmax(reshaped, dim=2)
        return softmax_output.unsqueeze(-1)
    else:
        # Use Triton kernel for large tensors
        # Launch the kernel
        high_dim_softmax_kernel[(total_programs,)](
            input_ptr=input_tensor.flatten().contiguous(),
            output_ptr=intermediate,
            batch_size=batch_size,
            flattened_dim=flattened_dim,
            in_channels=in_channels,
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS
        )
        
        # Normalize using Triton as well
        norm_tensor = torch.empty(batch_size, dtype=torch.float32, device=device)
        BLOCK_SIZE = 128
        norm_programs = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # For normalization, we need to compute row sums first
        reshaped_intermediate = intermediate.view(batch_size, flattened_dim)
        row_sums = torch.sum(reshaped_intermediate, dim=1)
        
        # Final normalization
        final_output = torch.empty_like(reshaped_intermediate)
        normalize_kernel[(norm_programs,)](
            intermediate_ptr=intermediate,
            final_output_ptr=final_output.flatten(),
            batch_size=batch_size,
            flattened_dim=flattened_dim,
            norm_ptr=row_sums,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Reshape to final output format
        return final_output.view(batch_size, 1, flattened_dim, 1)

@torch.fx.wrap
def replacement_kernel(conv_bias, conv_weight, input_tensor):
    """Optimized replacement function for high-dimensional attention"""
    batch_size, channels, height, width = input_tensor.shape
    flattened_dim = height * width
    total_elements = batch_size * flattened_dim
    return high_dim_attention_optimization(conv_bias, conv_weight, input_tensor, batch_size, flattened_dim, total_elements, channels)

def replacement_func():
    """Replacement function (MUST return function reference)"""
    return replacement_kernel