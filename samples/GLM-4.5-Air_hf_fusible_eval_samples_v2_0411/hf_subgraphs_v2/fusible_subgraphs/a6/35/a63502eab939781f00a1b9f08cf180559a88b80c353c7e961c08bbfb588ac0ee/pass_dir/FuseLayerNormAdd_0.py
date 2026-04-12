import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, weight, bias, residual):
    tmp_6 = torch.nn.functional.layer_norm(x, (weight.shape[0],), weight, bias, 1e-05)
    tmp_7 = residual + tmp_6
    return tmp_7

# Argument extraction function
def replacement_args(x, weight, bias, residual):
    return (x, weight, bias, residual)

# Optimized kernel for layer norm + add fusion
@triton.jit
def layer_norm_add_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    N, S, C,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one feature vector in one sequence position
    block_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Global position
    n = seq_id
    c_offset = block_id * BLOCK_SIZE_C
    
    # Compute bounds
    c_mask = c_offset + tl.arange(0, BLOCK_SIZE_C) < C
    
    # Load weight and bias once per block
    weight = tl.load(weight_ptr + c_offset, mask=c_mask, other=0.0)
    bias = tl.load(bias_ptr + c_offset, mask=c_mask, other=0.0)
    
    # Load input data
    x_ptr_base = x_ptr + n * S * C + c_offset
    residual_ptr_base = residual_ptr + n * S * C + c_offset
    
    x_data = tl.load(x_ptr_base + tl.arange(0, BLOCK_SIZE_C), mask=c_mask, other=0.0)
    residual_data = tl.load(residual_ptr_base + tl.arange(0, BLOCK_SIZE_C), mask=c_mask, other=0.0)
    
    # Mean calculation for layer normalization
    x_sum = tl.sum(x_data)
    x_mean = x_sum / C
    
    # Variance calculation
    x_var = tl.sum((x_data - x_mean) * (x_data - x_mean)) / C
    x_inv_std = 1.0 / tl.sqrt(x_var + eps)
    
    # Layer normalization: (x - mean) / std * weight + bias
    x_normalized = (x_data - x_mean) * x_inv_std
    x_result = x_normalized * weight + bias
    
    # Add residual
    out_data = x_result + residual_data
    
    # Store result
    out_ptr_base = out_ptr + n * S * C + c_offset
    tl.store(out_ptr_base + tl.arange(0, BLOCK_SIZE_C), out_data, mask=c_mask)

@torch.fx.wrap
def fused_layer_norm_add(x, weight, bias, residual):
    N, S, C = x.shape
    eps = 1e-05
    
    # Use smaller block size for better memory coalescing in feature dimension
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_N = 1
    
    # Calculate grid sizes
    num_blocks_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    num_blocks_n = N
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Launch kernel
    layer_norm_add_kernel[(num_blocks_c, num_blocks_n)](
        x,
        weight,
        bias,
        residual,
        output,
        N, S, C,
        eps,
        BLOCK_SIZE_C,
        BLOCK_SIZE_N
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_layer_norm_add