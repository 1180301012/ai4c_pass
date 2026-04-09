import torch
import triton
import triton.language as tl

# Pattern 2: Chained linear + multiplication (transformers pattern)
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = in_2 * linear
    return tmp_2

# Argument extraction for Pattern 2
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel for fused linear + multiplication operation
@triton.jit
def fused_linear_mul_kernel(
    weight_ptr,
    input_ptr,
    scale_ptr,
    output_ptr,
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    block_m_offset = (pid % ((batch_size * seq_len + BLOCK_M - 1) // BLOCK_M)) * BLOCK_M
    
    m_indices = block_m_offset + tl.arange(0, BLOCK_M)
    n_indices = tl.arange(0, BLOCK_N)
    k_indices = tl.arange(0, BLOCK_K)
    
    mask_m = m_indices < (batch_size * seq_len)
    mask_n = n_indices < out_features
    
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiplication: output = input @ weight.T
    for k in range(0, in_features, BLOCK_K):
        k_mask = k + k_indices < in_features
        
        # Input data loading
        input_ptrs = input_ptr + m_indices[:, None] * in_features + k + k_indices[None, :]
        input_data = tl.load(input_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        
        # Weight data loading
        weight_ptrs = weight_ptr + n_indices[:, None] * in_features + k + k_indices[None, :]
        weight_data = tl.load(weight_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)
        
        # Matrix multiplication with higher precision for accuracy
        accum += tl.dot(input_data, weight_data.to(tl.float32), out_features=BLOCK_K)
    
    # Convert back to original precision
    linear_result = accum.to(tl.float16)
    
    # Element-wise multiplication with scale
    scale_data = tl.load(scale_ptr + m_indices % (scale_ptr.shape[0] if len(scale_ptr.shape) > 0 else 1), 
                        mask=mask_m, other=0.0)
    scale_expanded = scale_data[:, None] if len(scale_ptr.shape) > 0 else scale_data
    
    fused_result = linear_result * scale_expanded
    
    # Store final result
    output_ptrs = output_ptr + m_indices[:, None] * out_features + n_indices[None, :]
    tl.store(output_ptrs, fused_result, mask=mask_m[:, None] & mask_n[None, :])

@torch.fx.wrap
def optimized_fused_linear_mul(in_0, in_1, in_2):
    # Handle different input tensor shapes
    if in_1.dim() == 3:
        batch_size, seq_len, in_features = in_1.shape
    else:
        batch_size = 1
        seq_len = in_1.shape[0] if len(in_1.shape) > 1 else 1
        in_features = in_1.shape[-1]
    
    out_features = in_0.shape[0]  # in_0 is weight matrix [out_features, in_features]
    
    # Handle scale tensor shape
    if len(in_2.shape) == 1:
        # Scale is [out_features] or [in_features], need to broadcast
        if in_2.shape[0] == out_features:
            scale_shape = (batch_size, seq_len, out_features)
        else:
            scale_shape = (batch_size, seq_len, in_features)
    else:
        scale_shape = in_2.shape
    
    output_shape = (batch_size, seq_len, out_features)
    fused_output = torch.empty(output_shape, device=in_1.device, dtype=in_1.dtype)
    
    # Optimized block sizes for better GPU utilization
    BLOCK_M = 128  # Process multiple rows at once
    BLOCK_N = 256  # Multiple columns
    BLOCK_K = 32   # K dimension tile size
    
    grid_m = (batch_size * seq_len + BLOCK_M - 1) // BLOCK_M
    grid_n = (out_features + BLOCK_N - 1) // BLOCK_N
    total_blocks = grid_m * grid_n
    grid = (total_blocks,)
    
    fused_linear_mul_kernel[grid](
        in_0, in_1, in_2, fused_output,
        batch_size, seq_len, in_features, out_features,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return (fused_output,)

def replacement_func():
    return optimized_fused_linear_mul