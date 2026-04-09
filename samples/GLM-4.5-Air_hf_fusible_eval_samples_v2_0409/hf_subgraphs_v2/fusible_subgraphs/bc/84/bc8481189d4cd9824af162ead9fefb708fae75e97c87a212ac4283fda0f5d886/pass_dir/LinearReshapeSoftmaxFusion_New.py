import torch
import triton
import triton.language as tl

# Pattern matching function for linear + reshape + softmax fusion
def pattern(x, weight, bias):
    """Match linear + reshape + softmax pattern with specific reshape shape [-1, 9, 1]"""
    linear = torch.nn.functional.linear(x, weight, bias)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4

# Argument extraction function
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Simple optimized linear operation using Triton
@triton.jit
def simple_linear_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_features, n_output, batch_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """Simple linear kernel using Triton"""
    pid_m = tl.program_id(0)
    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_range < batch_size
    
    # Process each m position
    for m_idx in range(BLOCK_SIZE_M):
        if (pid_m * BLOCK_SIZE_M + m_idx) < batch_size:
            # Matrix-vector multiplication: x[m_pos] @ weight.t() + bias
            m_pos = pid_m * BLOCK_SIZE_M + m_idx
            acc = tl.zeros(n_output, dtype=tl.float32)
            
            # Loop over features dimension
            for k in range(0, n_features, BLOCK_SIZE_K):
                k_range = k + tl.arange(0, BLOCK_SIZE_K)
                k_mask = k_range < n_features
                
                # Load x slice and weight block
                x_slice = tl.load(x_ptr + m_pos * n_features + k_range, 
                                mask=k_mask, other=0.0).to(tl.float32)
                weight_block = tl.load(weight_ptr + k_range[:, None] * n_output + tl.arange(0, n_output)[None, :], 
                                     mask=k_mask[:, None], other=0.0).to(tl.float32)
                
                # Accumulate: x @ weight
                acc += tl.sum(x_slice[:, None] * weight_block, axis=0, dtype=tl.float32)
            
            # Add bias and store result
            bias_vec = tl.load(bias_ptr + tl.arange(0, n_output), mask=tl.arange(0, n_output) < n_output, other=0.0)
            linear_out = acc + bias_vec
            
            # Store the result
            output_idx = (pid_m * BLOCK_SIZE_M + m_idx) * n_output
            for i in range(n_output):
                tl.store(out_ptr + output_idx + i, linear_out[i])

# Optimized reshape operation
@triton.jit  
def simple_reshape_kernel(
    input_ptr, output_ptr,
    input_size, output_size,
    BLOCK_SIZE: tl.constexpr
):
    """Simple reshape kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size
    
    # Load input data and store to output
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, input_data, mask=mask)

# Optimized softmax operation
@triton.jit
def simple_softmax_kernel(
    input_ptr, output_ptr,
    n_elements, dim_size,
    BLOCK_SIZE: tl.constexpr
):
    """Simple softmax kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Find position and compute softmax
    element_idx = offsets % dim_size
    group_idx = offsets // dim_size
    
    # Apply softmax within each group
    max_val = tl.max(input_data)
    shifted = input_data - max_val
    exp_vals = tl.exp(shifted)
    sum_exp = tl.sum(exp_vals)
    softmax_vals = exp_vals / (sum_exp + 1e-20)
    
    # Store result
    tl.store(output_ptr + offsets, softmax_vals, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_fused_linear_reshape_softmax(x, weight, bias):
    """Optimized fused linear + reshape + softmax operation using Triton"""
    x_shape = x.shape
    batch_size = x_shape[0] if len(x_shape) > 1 else 1
    if len(x_shape) == 1:
        x = x[None, :]  # Add batch dimension
    
    n_features = x.shape[-1]
    n_output = weight.shape[0]
    
    # Step 1: Compute linear output using Triton
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 64
    num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    linear_output = torch.empty(batch_size, n_output, dtype=x.dtype, device=x.device)
    simple_linear_kernel[(num_blocks_m,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=linear_output,
        n_features=n_features,
        n_output=n_output,
        batch_size=batch_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Step 2: Reshape to [-1, 9, 1] pattern
    # Linear output [batch_size, n_output] -> [batch_size, n_output//9, 9, 1]
    if n_output % 9 == 0:
        reshape_dim = n_output // 9
        reshaped_output = linear_output.reshape(batch_size, reshape_dim, 9, 1)
        
        # For softmax on dim=1, we need [batch_size*reshape_dim, 9, 1] -> flatten to [batch_size*reshape_dim*9]
        flat_input = reshaped_output.flatten()
        flat_size = flat_input.numel()
        
        # Step 3: Apply softmax using Triton on dim=1 (which corresponds to the 9 dimension)
        BLOCK_SIZE_SOFTMAX = 256
        num_blocks_softmax = (flat_size + BLOCK_SIZE_SOFTMAX - 1) // BLOCK_SIZE_SOFTMAX
        
        softmax_output = torch.empty_like(flat_input)
        simple_softmax_kernel[(num_blocks_softmax,)](
            input_ptr=flat_input,
            output_ptr=softmax_output,
            n_elements=flat_size,
            dim_size=9,
            BLOCK_SIZE=BLOCK_SIZE_SOFTMAX
        )
        
        # Reshape back to expected format
        return softmax_output.reshape(batch_size, reshape_dim, 9, 1)
    else:
        # Fallback: apply softmax on the entire linear output
        flat_size = linear_output.numel()
        BLOCK_SIZE_SOFTMAX = 256
        num_blocks_softmax = (flat_size + BLOCK_SIZE_SOFTMAX - 1) // BLOCK_SIZE_SOFTMAX
        
        softmax_output = torch.empty_like(linear_output)
        simple_softmax_kernel[(num_blocks_softmax,)](
            input_ptr=linear_output.flatten(),
            output_ptr=softmax_output.flatten(),
            n_elements=flat_size,
            dim_size=batch_size,
            BLOCK_SIZE=BLOCK_SIZE_SOFTMAX
        )
        
        return softmax_output.reshape(linear_output.shape)

# Replacement function
def replacement_func():
    return optimized_fused_linear_reshape_softmax