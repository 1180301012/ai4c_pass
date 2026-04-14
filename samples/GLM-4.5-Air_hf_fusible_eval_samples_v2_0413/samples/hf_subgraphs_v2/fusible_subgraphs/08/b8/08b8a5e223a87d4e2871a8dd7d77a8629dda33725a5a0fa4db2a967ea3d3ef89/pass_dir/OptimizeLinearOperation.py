import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_3):
    """
    Match linear operation: torch.nn.functional.linear(input, weight, bias)
    """
    linear = torch.nn.functional.linear(in_3, in_0, None)
    return linear

# Argument extraction function
def replacement_args(in_0, in_3):
    return (in_0, in_3)

# Optimized Linear operation using Triton
@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    seq_len, 
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized matrix multiplication kernel for linear transformation
    Input shape: [batch_size, seq_len, in_features]
    Weight shape: [in_features, out_features]
    Output shape: [batch_size, seq_len, out_features]
    """
    program_id = tl.program_id(axis=0)
    batch_id = program_id // seq_len
    seq_id = program_id % seq_len
    
    # Compute pointers for current batch and sequence position
    x_base_ptr = x_ptr + (batch_id * seq_len + seq_id) * in_features
    
    # Iterate over output features in blocks
    for k_start in range(0, out_features, BLOCK_SIZE_N):
        k_end = min(k_start + BLOCK_SIZE_N, out_features)
        
        # Load weights for current block
        w_ptrs = w_ptr + tl.arange(0, BLOCK_SIZE_K).to(tl.int32) * in_features
        w_block = tl.load(w_ptrs + k_start * in_features, mask=(tl.arange(0, BLOCK_SIZE_K) < in_features), other=0.0)
        
        # Initialize accumulator for current output block
        acc = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
        
        # Iterate over input features
        for i in range(0, in_features, BLOCK_SIZE_K):
            i_end = min(i + BLOCK_SIZE_K, in_features)
            
            # Load input values
            x_ptrs = x_base_ptr + i
            x_vals = tl.load(x_ptrs, mask=(tl.arange(0, i_end - i) < (i_end - i)), other=0.0)
            
            # Compute partial dot product
            acc += x_vals * w_block.to(tl.float32)
        
        # Store result for current output block
        out_base_ptr = out_ptr + (batch_id * seq_len + seq_id) * out_features
        out_ptrs = out_base_ptr + k_start
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=(tl.arange(0, k_end - k_start) < (k_end - k_start)))

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_linear(w, x):
    """Optimized linear operation using Triton"""
    batch_size, seq_len, in_features = x.shape
    out_features = w.shape[0]
    
    # Output shape: [batch_size, seq_len, out_features]
    out = torch.empty((batch_size, seq_len, out_features), dtype=torch.bfloat16, device=x.device)
    
    # Triton kernel launch
    grid_size = batch_size * seq_len
    
    linear_kernel[grid_size](
        x_ptr=x,
        w_ptr=w.t(),  # Transpose weight for column-major access
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len, 
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE_M=1,  # Process one seq at a time
        BLOCK_SIZE_N=32,  # Output features per thread
        BLOCK_SIZE_K=64,  # Input features per thread
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_linear