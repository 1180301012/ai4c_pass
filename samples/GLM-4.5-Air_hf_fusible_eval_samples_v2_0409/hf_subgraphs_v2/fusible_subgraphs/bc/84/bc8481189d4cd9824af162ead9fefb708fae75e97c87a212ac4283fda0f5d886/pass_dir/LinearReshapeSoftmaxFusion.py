import torch
import triton
import triton.language as tl
import math

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

# Triton kernel for fused linear + reshape + softmax
@triton.jit
def fused_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_features, n_output, batch_size, total_elements,
    reshape_dim1, reshape_dim2,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_SOFTMAX: tl.constexpr
):
    """Fused kernel: Linear + Reshape + Softmax"""
    pid_m = tl.program_id(0)
    pid_softmax = tl.program_id(1)
    
    # Compute ranges for linear operation
    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_range < batch_size
    
    # Compute ranges for softmax operation  
    softmax_range = pid_softmax * BLOCK_SIZE_SOFTMAX + tl.arange(0, BLOCK_SIZE_SOFTMAX)
    softmax_mask = softmax_range < reshape_dim1
    
    # Initialize output storage for softmax
    softmax_out = tl.zeros((BLOCK_SIZE_M, reshape_dim2), dtype=tl.float32)
    
    # Process each m position
    for m_idx in range(tl.minimum(BLOCK_SIZE_M, batch_size - pid_m * BLOCK_SIZE_M)):
        m_pos = pid_m * BLOCK_SIZE_M + m_idx
        
        # Compute linear output for this m position
        acc = tl.zeros(n_output, dtype=tl.float32)
        
        # Matrix-vector multiplication: x[m_pos] @ weight.t() + bias
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
        
        # Add bias
        bias_vec = tl.load(bias_ptr + tl.arange(0, n_output), mask=tl.arange(0, n_output) < n_output, other=0.0)
        linear_out = acc + bias_vec
        
        # Reshape linear output [1, n_output] -> [reshape_dim1, reshape_dim2]
        # For reshape_dim2 != 1 case, we take contiguous groups
        for pos in range(n_output):
            original_pos = pos * reshape_dim2
            if original_pos < n_output:
                # Take min of available elements or reshape_dim2 count
                count = tl.minimum(reshape_dim2, n_output - original_pos)
                
                # Reshape and apply softmax across reshape_dim1 dimension
                reshaped_vals = tl.zeros(reshape_dim1, dtype=tl.float32)
                for group_idx in range(count):
                    if m_idx < BLOCK_SIZE_M and (pos + group_idx) < reshape_dim1:
                        reshaped_vals[pos + group_idx] = linear_out[original_pos + group_idx]
                
                # Apply softmax across reshape_dim1
                if m_idx < BLOCK_SIZE_M:
                    # Find max for numerical stability
                    max_val = tl.max(reshaped_vals)
                    shifted = reshaped_vals - max_val
                    exp_vals = tl.exp(shifted)
                    sum_exp = tl.sum(exp_vals)
                    softmax_vals = exp_vals / (sum_exp + 1e-20)  # Add epsilon for stability
                    
                    # Store softmax results
                    for idx in range(reshape_dim1):
                        if idx < reshape_dim1:
                            softmax_out[m_idx, idx] = softmax_vals[idx]
    
    # Store final output
    for m_idx in range(BLOCK_SIZE_M):
        m_pos = pid_m * BLOCK_SIZE_M + m_idx
        for softmax_idx in range(BLOCK_SIZE_SOFTMAX):
            global_idx = m_pos * reshape_dim1 * reshape_dim2 + softmax_idx * reshape_dim2
            local_softmax_idx = softmax_idx
            
            if m_pos < batch_size and local_softmax_idx < reshape_dim1:
                for dim2_idx in range(reshape_dim2):
                    global_idx2 = global_idx + dim2_idx
                    if global_idx2 < total_elements:
                        tl.store(out_ptr + global_idx2, softmax_out[m_idx, local_softmax_idx] if dim2_idx == 0 else 0.0)

# Kernel wrapper
@torch.fx.wrap
def fused_linear_reshape_softmax(x, weight, bias, shape):
    """Fused linear + reshape + softmax operation using Triton"""
    x_shape = x.shape
    batch_size = x_shape[0] if len(x_shape) > 1 else 1
    if len(x_shape) == 1:
        x = x[None, :]  # Add batch dimension
    
    n_features = x.shape[-1]
    n_output = weight.shape[0]
    
    # Determine reshape dimensions
    reshape_dim1, reshape_dim2 = shape[-2:] if len(shape) > 2 else (shape[0], 1)
    total_elements = batch_size * reshape_dim1 * reshape_dim2
    
    # Verify reshape is valid
    if n_output != reshape_dim1 * reshape_dim2:
        raise ValueError(f"Linear output size {n_output} cannot be reshaped to [{reshape_dim1}, {reshape_dim2}]")
    
    # Set optimal block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = n_output
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_SOFTMAX = 32
    
    # Compute grid size
    num_blocks_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_softmax = (reshape_dim1 + BLOCK_SIZE_SOFTMAX - 1) // BLOCK_SIZE_SOFTMAX
    
    # Allocate output
    output = torch.empty(batch_size, reshape_dim1, reshape_dim2, dtype=x.dtype, device=x.device)
    
    # Launch fused kernel
    fused_kernel[(num_blocks_m, num_blocks_softmax)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output.flatten(),
        n_features=n_features,
        n_output=n_output,
        batch_size=batch_size,
        total_elements=total_elements,
        reshape_dim1=reshape_dim1,
        reshape_dim2=reshape_dim2,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_SOFTMAX=BLOCK_SIZE_SOFTMAX
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_linear_reshape_softmax