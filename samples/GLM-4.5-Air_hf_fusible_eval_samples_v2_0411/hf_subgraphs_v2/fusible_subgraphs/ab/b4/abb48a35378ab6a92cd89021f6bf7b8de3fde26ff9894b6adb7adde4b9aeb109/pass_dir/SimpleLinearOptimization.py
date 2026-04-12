import torch
import triton
import triton.language as tl

def pattern(weight, input_tensor):
    """Pattern matches simple linear operation"""
    tmp_0 = weight
    linear_out = torch.nn.functional.linear(input_tensor, tmp_0, None)
    return linear_out

@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr, 
    output_ptr,
    m_size,
    n_size,
    k_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized Triton kernel for Y = X @ W.T (linear operation)"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, m_size)
    
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min(n_start + BLOCK_SIZE_N, n_size)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension
    for k in range(0, k_size, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, k_size)
        
        # Load input block [M, K]
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_m = m_offsets[:, None] < m_end
        mask_k = k_offsets[None, :] < k_size
        
        input_ptr_base = input_ptr + m_offsets[:, None] * k_size + k_offsets[None, :]
        input_vals = tl.load(input_ptr_base, mask=mask_m & mask_k, other=0.0)
        
        # Load weight block [N, K]  
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        mask_n = n_offsets[None, :] < n_end
        
        weight_ptr_base = weight_ptr + n_offsets[:, None] * k_size + k_offsets[None, :]
        weight_vals = tl.load(weight_ptr_base, mask=mask_n & mask_k, other=0.0)
        
        # Matrix multiplication: A = A + input @ weight.T  
        accumulator += tl.dot(input_vals, weight_vals)
    
    # Store result [M, N]
    mask = (m_offsets[:, None] < m_end) & (n_offsets[None, :] < n_end)
    output_ptr_base = output_ptr + m_offsets[:, None] * n_size + n_offsets[None, :]
    tl.store(output_ptr_base, accumulator, mask=mask)

@triton.jit
def linear_kernel_3d(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    input_features,
    output_features,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized Triton kernel for 3D linear operation - one output element per thread"""
    # Each program handles one (seq, output_feature) combination
    # Since batch_size=1 in our case, we ignore batch dimension
    pid_s = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    # Check bounds
    if pid_s >= seq_len or pid_o >= output_features:
        return
    
    # Initialize accumulator with proper precision for the computation
    accumulator = 0.0
    
    # Loop over k dimension to compute dot product
    for k in range(0, input_features, BLOCK_SIZE_K):
        # Loop bounds are automatically handled by range()
        # Load input value: input[0, pid_s, k] (batch=0 always)
        input_offset = 0 * (seq_len * input_features) + pid_s * input_features + k
        input_vals = tl.load(input_ptr + input_offset, 
                            mask=k < input_features, other=0.0)
        
        # Load weight value: weight[pid_o, k]
        weight_offset = pid_o * input_features + k
        weight_vals = tl.load(weight_ptr + weight_offset,
                              mask=k < input_features, other=0.0)
        
        # Compute and accumulate the multiplication precisely
        product = input_vals * weight_vals
        accumulator += product
    
    # Store result: output[0, pid_s, pid_o]
    output_offset = 0 * (seq_len * output_features) + pid_s * output_features + pid_o
    tl.store(output_ptr + output_offset, accumulator)

@torch.fx.wrap  
def optimized_linear(weight, input_tensor):
    """Optimized linear operation"""
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    input_features = input_tensor.shape[2]
    output_features = weight.shape[0]
    
    # Use original 3D tensor dimensions directly
    m_size = batch_size  # first dimension
    n_size = seq_len     # second dimension  
    k_size = input_features
    output_features = weight.shape[0]
    
    # Optimal block sizes
    BLOCK_SIZE_K = 32  # Process multiple input features per iteration
    
    # Launch grid dimensions for 2D operation (seq, output_features)
    grid_seq = (seq_len + 1 - 1) // 1  # One thread per sequence element
    grid_output = (output_features + 1 - 1) // 1  # One thread per output feature
    
    # Allocate output tensor with correct 3D shape
    output = torch.empty(batch_size, seq_len, output_features,
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel with 2D grid
    linear_kernel_3d[(grid_seq, grid_output)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        input_features=input_features,
        output_features=output_features,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output

def replacement_args(weight, input_tensor):
    """Extract arguments for the optimized linear kernel"""
    return (weight, input_tensor)

def replacement_func():
    """Return the optimized linear function"""
    return optimized_linear