import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the computation in model.py
def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return (tmp_3,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel for float16 data type - simple vectorized version
@triton.jit
def fused_fp16_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    batch_size,
    dim_to_sum,
    dim3_size,
    dim2_size,
    BLOCK_SIZE_DIM1: tl.constexpr,
):
    # Each program handles one (batch, height, width) combination
    pid = tl.program_id(0)
    
    if pid >= batch_size * dim2_size * dim3_size:
        return
    
    # Decompose program ID into batch, height, width coordinates
    batch_idx = pid // (dim2_size * dim3_size)
    hw_idx = pid % (dim2_size * dim3_size)
    height_idx = hw_idx // dim3_size
    width_idx = hw_idx % dim3_size
    
    # Load and multiply elements for the entire dim_to_sum dimension
    sum_value = 0.0
    
    # Process elements in blocks for better memory coalescing
    for dim1_offset in range(0, dim_to_sum, BLOCK_SIZE_DIM1):
        # Load a block of elements for in_0 and in_1
        offsets = dim1_offset + tl.arange(0, BLOCK_SIZE_DIM1)
        
        # Calculate memory addresses for this block
        batch_offset = batch_idx * dim_to_sum * dim2_size * dim3_size
        element_offsets = batch_offset + offsets * dim2_size * dim3_size + height_idx * dim3_size + width_idx
        
        # Load with bounds checking
        mask = offsets < dim_to_sum
        in_0_block = tl.load(in_0_ptr + element_offsets, mask=mask, other=0.0)
        in_1_block = tl.load(in_1_ptr + element_offsets, mask=mask, other=0.0)
        
        # Multiply and sum (this happens in parallel for the whole block)
        product_block = in_0_block * in_1_block
        sum_value += tl.sum(product_block)
    
    # Apply sigmoid activation
    result = 1.0 / (1.0 + tl.exp(-sum_value))
    
    # Store result - output has shape [batch_size, 1, dim2_size, dim3_size]
    # For float16, we need to cast the result back
    result_fp16 = tl.cast(result, tl.float16)
    
    # Calculate output index
    output_idx = pid
    tl.store(out_ptr + output_idx, result_fp16)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_fp16_computation(in_0, in_1):
    batch_size, dim_to_sum, dim2_size, dim3_size = in_0.shape
    
    # Create output tensor with shape [batch_size, 1, dim2_size, dim3_size]
    # This matches torch.sum(..., dim=1).unsqueeze(1) result
    out_shape = (batch_size, 1, dim2_size, dim3_size)
    out = torch.empty(out_shape, dtype=torch.float16, device=in_0.device)
    
    # Flatten output for kernel access
    flattened_out = out.view(-1)
    
    # Calculate total number of (batch, height, width) combinations
    total_hw_elements = batch_size * dim2_size * dim3_size
    
    # Use optimal block size for vectorized memory access
    # For dim_to_sum dimension (typically 64): use moderate block size for good parallelism
    BLOCK_SIZE_DIM1 = 128
    
    # Launch kernel with 1D grid - each program handles one (batch, height, width) combo
    fused_fp16_kernel[(total_hw_elements,)](
        in_0,
        in_1,
        flattened_out,
        batch_size,
        dim_to_sum,
        dim3_size,
        dim2_size,
        BLOCK_SIZE_DIM1=BLOCK_SIZE_DIM1,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_fp16_computation