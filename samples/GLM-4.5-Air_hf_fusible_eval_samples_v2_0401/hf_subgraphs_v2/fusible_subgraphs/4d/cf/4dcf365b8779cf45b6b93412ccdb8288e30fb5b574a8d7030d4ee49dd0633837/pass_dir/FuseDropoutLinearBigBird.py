import torch
import triton
import triton.language as tl

@triton.jit
def fused_dropout_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_m,
    input_k,
    output_k,
    dropout_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Compute mask
    m_mask = m_offsets < input_m
    n_mask = n_offsets < output_k
    
    # Initialize accumulator for this C tile
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, input_k, BLOCK_K):
        # Compute K range for this iteration
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < input_k
        
        # Load input tile: A[m, k] and cast to fp32
        a = tl.load(
            input_ptr + (m_offsets[:, None] * input_k + k_offsets[None, :]),
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Load weight tile: B[k, n] and cast to fp32
        b = tl.load(
            weight_ptr + (k_offsets[:, None] * output_k + n_offsets[None, :]),
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        # Scale input by dropout factor (fused)
        a_scaled = a * dropout_scale
        
        # Multiply and accumulate
        accumulator += tl.dot(a_scaled, b)
    
    # Load bias and add it
    bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
    output = accumulator + bias[None, :]
    
    # Store result
    output_ptrs = output_ptr + (m_offsets[:, None] * output_k + n_offsets[None, :])
    tl.store(output_ptrs, output, mask=m_mask[:, None] & n_mask[None, :])

@torch.fx.wrap
def fused_dropout_linear(input, weight, bias):
    # Get input dimensions and transform to 2D
    if input.dim() == 3:
        input_2d = input.reshape(-1, input.size(-1))
        original_shape = input.shape
        output_shape = original_shape[:-1] + (bias.size(0),)
    else:
        input_2d = input
        original_shape = input.shape
        output_shape = original_shape[:-1] + (bias.size(0),)
    
    # Get dimensions
    m, k = input_2d.shape
    n = bias.size(0)
    
    # For BigBird dropout with p=0.1, training=False
    dropout_scale = 0.9
    
    # Create output tensor
    output = torch.empty((m, n), dtype=torch.float32, device=input.device)
    
    # Choose block sizes based on tensor sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    # Calculate grid size
    grid_m = (m + BLOCK_M - 1) // BLOCK_M
    grid_n = (n + BLOCK_N - 1) // BLOCK_N
    
    # Launch kernel
    fused_dropout_linear_kernel[(grid_m, grid_n)](
        input_2d,
        weight,
        bias,
        output,
        m, k, n,
        dropout_scale,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    # Reshape back to original if needed
    if len(original_shape) == 3:
        output = output.view(original_shape[:-1] + (n,))
    
    return output

def pattern(tmp_3, in_1, in_0):
    """Pattern: dropout(input, 0.1, False, False) followed by linear(dropout_result, weight, bias)"""
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear

def replacement_args(tmp_3, in_1, in_0):
    """Extract arguments: dropout input, weight, bias"""
    return (tmp_3, in_1, in_0)

def replacement_func():
    """Return fused dropout + linear kernel"""
    return fused_dropout_linear