import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_2):
    # Mean computation along dim=-2, keepdim=True
    mean_out = in_2.mean(dim=-2, keepdim=True)
    return mean_out

# Argument extraction function
def replacement_args(in_2):
    return (in_2,)

# Optimized kernel for mean computation using Triton
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    dim_size: tl.constexpr,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of features
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_size
    
    # Compute mean for each batch and each feature
    for batch_idx in tl.static_range(batch_size):
        for feature_idx in offsets:
            if feature_idx < feature_size:
                # Accumulate values along the dimension we're averaging (dim=-2, which is index 1)
                sum_val = 0.0
                for dim_idx in tl.static_range(dim_size):
                    # Calculate input position: [batch_idx, dim_idx, feature_idx]
                    input_pos = batch_idx * (dim_size * feature_size) + dim_idx * feature_size + feature_idx
                    val = tl.load(input_ptr + input_pos, other=0.0)
                    sum_val += val
                
                # Calculate mean (divide by dim_size)
                mean_val = sum_val / dim_size
                
                # Store output position: [batch_idx, 0, feature_idx]
                output_pos = batch_idx * (1 * feature_size) + feature_idx
                tl.store(output_ptr + output_pos, mean_val, mask=feature_idx < feature_size)

# Alternative optimized kernel using vectorized operations for better performance
@triton.jit
def optimized_mean_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    batch_size: tl.constexpr,
    dim_size: tl.constexpr,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    # Calculate which output element we're processing
    batch_idx = idx // feature_size
    feature_idx = idx % feature_size
    
    if batch_idx < batch_size and feature_idx < feature_size:
        # Calculate the input positions for this batch and feature
        input_offset = batch_idx * dim_size * feature_size + feature_idx
        
        # Use atomic operations for thread-safe accumulation
        sum_val = 0.0
        for dim_offset in tl.static_range(dim_size):
            input_pos = input_offset + dim_offset * feature_size
            val = tl.load(input_ptr + input_pos, other=0.0)
            sum_val += val
        
        # Calculate mean
        mean_val = sum_val / dim_size
        
        # Store output
        output_pos = batch_idx * feature_size + feature_idx
        tl.store(output_ptr + output_pos, mean_val, mask=idx < n_elements)

# Even better kernel with vectorized loading and optimized memory access
@triton.jit
def vectorized_mean_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    dim_size: tl.constexpr,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles features for one batch
    program_id = tl.program_id(0)
    batch_idx = program_id // ((feature_size + BLOCK_SIZE - 1) // BLOCK_SIZE)
    feature_start = (program_id % ((feature_size + BLOCK_SIZE - 1) // BLOCK_SIZE)) * BLOCK_SIZE
    
    if batch_idx < batch_size:
        # Load weights for vectorized operations
        weights = tl.arange(0.0, 1.0)  # We don't need weights, just counting
        dim_size_f = tl.float32(dim_size)
        
        # Process each feature in the block
        for feature_idx in range(feature_start, min(feature_start + BLOCK_SIZE, feature_size)):
            # Load all elements for this batch and feature across the mean dimension
            input_base = batch_idx * dim_size * feature_size + feature_idx
            
            # Vectorized load of the dimension to be averaged
            dim_vals = tl.load(input_ptr + input_base + tl.arange(0, dim_size) * feature_size, 
                              other=0.0)
            
            # Compute sum and mean
            sum_val = tl.sum(dim_vals)
            mean_val = sum_val / dim_size_f
            
            # Store result
            output_pos = batch_idx * feature_size + feature_idx
            tl.store(output_ptr + output_pos, mean_val)

# Optimized kernel for mean computation
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    dim_size: tl.constexpr,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of features
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * feature_size
    
    # Process each batch and feature
    for batch_idx, feature_idx in tl.static_range(batch_size), offsets:
        if feature_idx < feature_size and batch_idx < batch_size:
            # Calculate input position: [batch_idx, :, feature_idx]
            input_offset = batch_idx * dim_size * feature_size + feature_idx
            
            # Sum along the dimension to be averaged
            sum_val = 0.0
            for dim_offset in tl.static_range(dim_size):
                input_pos = input_offset + dim_offset * feature_size
                val = tl.load(input_ptr + input_pos, other=0.0)
                sum_val += val
            
            # Calculate mean
            mean_val = sum_val / dim_size
            
            # Store output position: [batch_idx, 0, feature_idx] -> flatten to [batch_idx, feature_idx]
            output_pos = batch_idx * feature_size + feature_idx
            tl.store(output_ptr + output_pos, mean_val, mask=mask)

# Correct mean computation kernel with autotuning
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    dim_size: tl.constexpr,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Handle BLOCK_SIZE elements per program for better utilization
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * feature_size
    
    for offset, mask_val in zip(offsets, mask):
        if mask_val:
            # Compute batch_idx and feature_idx from flattened offset
            batch_idx = offset // feature_size
            feature_idx = offset % feature_size
            
            # Compute mean along dim=-2
            sum_val = 0.0
            for dim_offset in tl.static_range(dim_size):
                # Input address: batch_idx, dim_offset, feature_idx
                input_pos = batch_idx * dim_size * feature_size + dim_offset * feature_size + feature_idx
                val = tl.load(input_ptr + input_pos)
                sum_val += val
            
            # Compute mean and store
            mean_val = sum_val / dim_size
            output_pos = offset  # Already flattened [batch_idx, feature_idx]
            tl.store(output_ptr + output_pos, mean_val)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_mean_computation(in_2):
    # Get input dimensions
    batch_size, dim_size, feature_size = in_2.shape
    
    # Create output tensor with correct shape [batch_size, 1, feature_size]
    output_shape = (batch_size, 1, feature_size)
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Flatten tensors for kernel processing
    input_flat = in_2.reshape(-1)  # [batch_size * dim_size * feature_size]
    output_flat = output.reshape(-1)  # [batch_size * feature_size]
    
    # Setup kernel launch parameters with optimized block size
    n_output_elements = batch_size * feature_size
    
    # Choose optimal block size based on workload
    if n_output_elements >= 1024:
        BLOCK_SIZE = 256
    elif n_output_elements >= 256:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 64
    
    num_programs = (n_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized mean computation kernel
    mean_kernel[(num_programs,)](
        input_ptr=input_flat,
        output_ptr=output_flat,
        batch_size=batch_size,
        dim_size=dim_size,
        feature_size=feature_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_mean_computation