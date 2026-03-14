import torch
import triton
import triton.language as tl

# Pattern matching function - matches the softmax + dropout pattern
def pattern(in_0, in_1):
    # Match the exact computation from the model
    tmp_0 = torch.nn.functional.softmax(in_0, dim=-1)
    tmp_1 = torch.nn.functional.dropout(tmp_0, p=0.0, training=False)
    tmp_2 = torch.bmm(tmp_1, in_1)
    tmp_3 = tmp_2.view(1, -1, 1, -1)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.reshape(1, 1, -1)
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel that eliminates the dropout
@triton.jit
def optimized_softmax_bmm_kernel(
    attn_weights_ptr,
    value_states_ptr,
    output_ptr,
    batch_size,
    value_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Softmax kernel for attention weights
    pid_m = tl.program_id(0)
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_m = offset_m % batch_size
    
    # Load attention weights [batch_size, 1, 1]
    attn_weights = tl.load(attn_weights_ptr + offset_m * 1, mask=(offset_m < batch_size), other=0.0)
    
    # Apply softmax (exponential and normalization)
    max_val = tl.max(attn_weights, axis=0)
    exp_attn = tl.exp(attn_weights - max_val)
    sum_exp = tl.sum(exp_attn, axis=0)
    softmax_weights = exp_attn / sum_exp
    
    # BMM computation - softmax_weights @ value_states
    pid_n = tl.program_id(1)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_n = offset_n % value_dim
    
    # Load value states [batch_size, 1, value_dim]
    value_states = tl.load(value_states_ptr + offset_m * value_dim + offset_n, 
                          mask=(offset_m < batch_size)[:, None] & (offset_n[None, :] < value_dim), 
                          other=0.0)
    
    # Compute matrix multiplication
    output = tl.sum(softmax_weights * value_states, axis=0)
    
    # Store result
    tl.store(output_ptr + offset_m * value_dim + offset_n, output, 
             mask=(offset_m < batch_size)[:, None] & (offset_n[None, :] < value_dim))

@torch.fx.wrap
def optimized_softmax_bmm_forward(in_0, in_1):
    batch_size, _, _ = in_0.shape
    value_dim = in_1.shape[-1]
    
    # Reshape for output: [batch_size, value_dim]
    output_shape = (batch_size, value_dim)
    output = torch.zeros(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Grid configuration for Triton kernel
    BLOCK_SIZE_M = 32  # Process multiple rows at once
    BLOCK_SIZE_N = 256  # Process multiple columns at once
    BLOCK_SIZE_K = 1   # Attention weights are [1, 1]
    
    grid = (
        (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (value_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
    )
    
    optimized_softmax_bmm_kernel[grid](
        in_0,
        in_1,
        output,
        batch_size,
        value_dim,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    # Apply the same tensor transformations as the original
    tmp_3 = output.view(1, batch_size, 1, value_dim)
    tmp_4 = tmp_3.transpose(1, 2)
    result = tmp_4.reshape(1, 1, batch_size * value_dim)
    
    return result

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_softmax_bmm_forward