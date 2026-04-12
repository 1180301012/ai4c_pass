import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel
@triton.jit
def fused_mul_sum_unsqueeze_sigmoid_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    dim1_size,
    dim2_size,
    dim3_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program identifier for the fused kernel
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate the range for this program
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, dim2_size)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, dim3_size)
    
    # Shared memory for accumulation along dim1 (which is 64 in this case)
    shared_size = BLOCK_SIZE_M
    shared_mem = tl.arange(0, shared_size)
    
    # Process each batch
    for b in range(batch_size):
        # Initialize accumulator for each [dim2, dim3] position
        # We need to sum along dim1 (size 64) for each position
        accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        
        # Process chunks of dim1 to achieve full parallelism
        for chunk_start in range(0, dim1_size, BLOCK_SIZE_M):
            chunk_end = min(chunk_start + BLOCK_SIZE_M, dim1_size)
            chunk_size = chunk_end - chunk_start
            
            # Calculate offsets for all dimensions
            offsets_m = tl.arange(0, min(chunk_size, BLOCK_SIZE_M))
            offsets_n = tl.arange(0, BLOCK_SIZE_N)
            
            # Create coordinate matrices
            m_coords = offsets_m[:, None] + chunk_start  # [BLOCK_SIZE_M] -> [BLOCK_SIZE_M, 1]
            n_coords = offsets_n[None, :]  # [BLOCK_SIZE_N] -> [1, BLOCK_SIZE_N]
            
            # Calculate flattened indices for batch b and coordinates
            x_indices = (b * dim1_size * dim2_size * dim3_size + 
                        m_coords * dim2_size * dim3_size + 
                        tl.arange(0, dim2_size)[:, None] * dim3_size + 
                        n_coords)
            y_indices = (b * dim1_size * dim2_size * dim3_size + 
                        m_coords * dim2_size * dim3_size + 
                        tl.arange(0, dim2_size)[:, None] * dim3_size + 
                        n_coords)
            
            # Ensure we don't go out of bounds
            m_mask = m_coords < dim1_size
            n_mask = n_coords < dim3_size
            mask = m_mask[:, None] & n_mask
            
            # Load input values
            if chunk_size > 0:
                x_vals = tl.load(x_ptr + x_indices, mask=mask, other=0.0)
                y_vals = tl.load(y_ptr + y_indices, mask=mask, other=0.0)
                
                # Accumulate products
                products = x_vals * y_vals
                accumulator += products[:, :chunk_size]
        
        # Apply sigmoid to accumulated sums and store results
        sigmoid_results = 1.0 / (1.0 + tl.exp(-accumulator))
        
        # Calculate output indices
        out_m_coords = tl.arange(0, min(m_end - m_start, BLOCK_SIZE_M))
        out_n_coords = tl.arange(0, min(n_end - n_start, BLOCK_SIZE_N))
        out_m = out_m_coords[:, None] + m_start
        out_n = out_n_coords[None, :] + n_start
        
        out_indices = (b * dim2_size * dim3_size + 
                      tl.arange(0, batch_size)[:, None] * dim3_size + 
                      out_n)
    
    # Alternative simpler approach for the given tensor structure
    # For tensors of shape [batch, 64, 64, 64], we can optimize differently
    pass

# Simple but robust fused kernel
@triton.jit
def fused_mul_sum_unsqueeze_sigmoid_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    dim1_size,
    dim2_size,
    dim3_size,
    dtype: tl.constexpr,
):
    # Program identifiers - each program handles one [dim2, dim3] position
    pid_b = tl.program_id(0)      # batch dimension
    pid_m = tl.program_id(1)      # dim2 dimension  
    pid_n = tl.program_id(2)      # dim3 dimension
    
    # Initialize accumulator for sum along dimension 1 using high precision
    accumulator = 0.0
    
    # Sum along dimension 1 (index 1, size dim1_size) for current [pid_b, pid_m, pid_n] position
    for k in range(dim1_size):
        # Calculate absolute index for [batch, k, pid_m, pid_n]
        index = (pid_b * dim1_size * dim2_size * dim3_size + 
                k * dim2_size * dim3_size + 
                pid_m * dim3_size + pid_n)
        
        # Load values from both tensors
        x_val = tl.load(x_ptr + index, mask=True, other=0.0)
        y_val = tl.load(y_ptr + index, mask=True, other=0.0)
        
        # Add product to accumulator (use float32 for accumulation for better precision)
        accumulator += (x_val * y_val).to(tl.float32)
    
    # Apply sigmoid to final result using high precision
    result = 1.0 / (1.0 + tl.exp(-accumulator))
    
    # Cast back to original data type
    result = result.to(dtype)
    
    # Store result at [pid_b, 0, pid_m, pid_n] where output has shape [batch, 1, dim2, dim3]
    # The output tensor has only 1 element in dimension 1, so stride for dim1 is 1 * dim2 * dim3
    out_index = (pid_b * (1 * dim2_size * dim3_size) + 
                pid_m * dim3_size + pid_n)
    
    # Store the result
    tl.store(out_ptr + out_index, result)

# Optimized kernel wrapper with better program distribution
@torch.fx.wrap  
def fused_kernel_wrapper(in_0, in_1):
    # Get tensor shapes and properties
    batch_size, dim1_size, dim2_size, dim3_size = in_0.shape
    dtype = in_0.dtype
    
    # Correct output shape after sum(dim=1) and unsqueeze(1): [batch, 1, dim2, dim3]
    out_shape = (batch_size, 1, dim2_size, dim3_size)
    out = torch.empty(out_shape, dtype=dtype, device=in_0.device)
    
    # Convert dtype to Triton type constant
    if dtype == torch.float16:
        triton_dtype = tl.float16
    elif dtype == torch.bfloat16:
        triton_dtype = tl.bfloat16
    elif dtype == torch.float32:
        triton_dtype = tl.float32
    else:
        # Fallback to float32 for unsupported types
        triton_dtype = tl.float32
    
    # Use the full grid dimensions - one program per position for simplicity and correctness
    grid_x = dim2_size
    grid_y = dim3_size
    grid_z = batch_size
    
    # Launch the kernel with optimized program distribution
    fused_mul_sum_unsqueeze_sigmoid_kernel[(grid_z, grid_x, grid_y)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        dim1_size=dim1_size,
        dim2_size=dim2_size,
        dim3_size=dim3_size,
        dtype=triton_dtype,
    )
    
    return out

def replacement_func():
    return fused_kernel_wrapper