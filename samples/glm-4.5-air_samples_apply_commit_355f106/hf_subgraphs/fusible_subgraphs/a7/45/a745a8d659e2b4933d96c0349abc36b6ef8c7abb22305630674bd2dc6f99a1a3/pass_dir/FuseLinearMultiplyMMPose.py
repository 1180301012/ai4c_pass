import torch
import triton
import triton.language as tl

# Pattern matching function for MMPose: linear + multiply pattern  
def pattern(weight, scale, input_x, weight_matrix):
    """
    Pattern that matches: 
    tmp_0 = weight
    tmp_1 = scale  
    tmp_2 = input_x
    tmp_3 = weight_matrix
    tmp_4 = linear_transformation(tmp_3, tmp_0)  # Will be replaced
    tmp_5 = tmp_2 * tmp_1
    return (tmp_5, tmp_4)
    """
    # Create temporary variables matching the original computation
    tmp_0 = weight
    tmp_1 = scale
    tmp_2 = input_x
    tmp_3 = weight_matrix
    
    # This will be replaced by the fused kernel
    tmp_4 = tmp_3 * tmp_0  # Placeholder - will be replaced
    tmp_5 = tmp_2 * tmp_1
    
    return tmp_5, tmp_4

# Argument extraction function
def replacement_args(weight, scale, input_x, weight_matrix):
    return (weight, scale, input_x, weight_matrix)

# Optimized Triton kernel for fused linear + multiply
@triton.jit
def fused_linear_multiply_kernel(
    weight_ptr,           # Weight matrix [out_features, in_features]
    scale_ptr,            # Scale tensor [out_features] 
    input_x_ptr,          # Input tensor [batch, seq_len, out_features]
    weight_matrix_ptr,    # Weight matrix for linear [in_features,hidden]
    multiply_result_ptr,  # Output for element-wise multiply
    linear_result_ptr,    # Output for linear transformation
    batch_size,
    seq_len_out,
    in_features,
    hidden_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program id for batch and sequence dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Ensure we don't go out of bounds
    if pid_m >= batch_size or pid_n >= seq_len_out:
        return
    
    # Load scale for this position
    scale_val = tl.load(scale_ptr + pid_m * in_features + tl.arange(0, BLOCK_SIZE_N), 
                        mask=tl.arange(0, BLOCK_SIZE_N) < in_features, 
                        other=0.0)
    
    # Compute base offset for this batch and sequence position
    x_offset = (pid_m * seq_len_out + pid_n) * in_features
    
    # Load input_x data
    input_x = tl.load(input_x_ptr + x_offset + tl.arange(0, BLOCK_SIZE_N),
                      mask=tl.arange(0, BLOCK_SIZE_N) < in_features,
                      other=0.0)
    
    # Element-wise multiplication: input_x * scale
    multiply_out = input_x * scale_val
    
    # Store element-wise multiply result
    tl.store(multiply_result_ptr + x_offset + tl.arange(0, BLOCK_SIZE_N),
             multiply_out, mask=tl.arange(0, BLOCK_SIZE_N) < in_features)
    
    # Matrix multiplication for linear part
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k in range(0, hidden_features, BLOCK_SIZE_K):
        # Load weight matrix chunk
        weight_chunk = tl.load(weight_matrix_ptr + 
                              (pid_n * hidden_features + k) * in_features + 
                              tl.arange(0, BLOCK_SIZE_K),
                              mask=tl.arange(0, BLOCK_SIZE_K) < (hidden_features - k),
                              other=0.0)
        
        # Load weight matrix (for linear)
        weight_chunk_linear = tl.load(weight_ptr + 
                                     k * in_features + 
                                     tl.arange(0, BLOCK_SIZE_N),
                                     mask=tl.arange(0, BLOCK_SIZE_N) < in_features,
                                     other=0.0)
        
        # Matrix multiplication accumulator
        acc += weight_chunk * tl.sum(weight_chunk_linear[None, :] * input_x[:, None], axis=1)
    
    # Store linear result
    linear_out_offset = (pid_m * seq_len_out + pid_n) * hidden_features
    tl.store(linear_result_ptr + linear_out_offset + tl.arange(0, BLOCK_SIZE_N),
             acc.to(tl.float32), mask=tl.arange(0, BLOCK_SIZE_N) < hidden_features)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_linear_multiply_mmpose(weight, scale, input_x, weight_matrix):
    # Get tensor shapes
    batch_size = input_x.size(0)
    seq_len_out = input_x.size(1)
    in_features = input_x.size(2)
    hidden_features = weight_matrix.size(2)
    
    # Create output tensors
    multiply_result = torch.empty_like(input_x)
    linear_result = torch.empty((batch_size, seq_len_out, hidden_features), 
                               dtype=input_x.dtype, device=input_x.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_M),
            triton.cdiv(seq_len_out, BLOCK_SIZE_N))
    
    fused_linear_multiply_kernel[grid](
        weight,
        scale, 
        input_x,
        weight_matrix,
        multiply_result,
        linear_result,
        batch_size,
        seq_len_out,
        in_features,
        hidden_features,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return multiply_result, linear_result

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_linear_multiply_mmpose