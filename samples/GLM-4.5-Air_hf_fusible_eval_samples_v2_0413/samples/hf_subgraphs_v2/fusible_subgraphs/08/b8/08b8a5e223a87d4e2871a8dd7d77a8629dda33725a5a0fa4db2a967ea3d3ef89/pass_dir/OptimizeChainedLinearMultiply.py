import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """
    Match chained linear + multiplication pattern:
    linear = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2 = in_1 * linear
    Return both intermediate linear result and final multiplication result
    """
    linear = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2 = in_1 * linear
    return linear, tmp_2

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Linear + Multiply using Triton
@triton.jit
def fused_linear_mul_kernel(
    x_ptr,        # Input tensor [B, S, D_in]
    w_ptr,        # Weight [D_in, D_out] (transposed)
    y_ptr,        # Multiply tensor [B, S, D_out]
    out_ptr,      # Output [B, S, D_out]
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused linear + multiplication kernel for transformer patterns
    First computes linear: x @ w.t()
    Then multiplies by y: y * (x @ w.t())
    """
    program_id = tl.program_id(axis=0)
    batch_id = program_id // seq_len
    seq_id = program_id % seq_len
    
    # Pointer for current position in input
    x_base_ptr = x_ptr + (batch_id * seq_len + seq_id) * in_features
    y_base_ptr = y_ptr + (batch_id * seq_len + seq_id) * out_features
    out_base_ptr = out_ptr + (batch_id * seq_len + seq_id) * out_features
    
    # Initialize accumulator for linear output
    linear_acc = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    
    # Step 1: Compute linear transformation (x @ w.t.)
    for k in range(0, out_features, BLOCK_SIZE_N):
        k_end = min(k + BLOCK_SIZE_N, out_features)
        
        # Load weights for current output block
        w_ptrs = w_ptr + tl.arange(0, BLOCK_SIZE_K).to(tl.int32) * in_features
        w_block = tl.load(w_ptrs + k * in_features, mask=(tl.arange(0, BLOCK_SIZE_K) < in_features), other=0.0)
        w_block = w_block.to(tl.float32)
        
        # Reset accumulator for this output block
        acc = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
        
        # Compute dot product over input features
        for i in range(0, in_features, BLOCK_SIZE_K):
            i_end = min(i + BLOCK_SIZE_K, in_features)
            
            # Load input values
            x_ptrs = x_base_ptr + i
            x_vals = tl.load(x_ptrs, mask=(tl.arange(0, i_end - i) < (i_end - i)), other=0.0)
            x_vals = x_vals.to(tl.float32)
            
            # Accumulate partial dot product
            acc += x_vals * w_block
        
        # Store linear output temporarily
        out_ptrs = out_base_ptr + k
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=(tl.arange(0, k_end - k) < (k_end - k)))
    
    # Step 2: Load y values and perform multiplication
    # Load all y values for current position (can be optimized further)
    y_vals = tl.load(y_base_ptr, mask=(tl.arange(0, out_features) < out_features), other=0.0)
    
    # Multiply linear result by y values
    for k in range(0, out_features, BLOCK_SIZE_N):
        k_end = min(k + BLOCK_SIZE_N, out_features)
        
        # Load linear output for this block
        linear_ptrs = out_base_ptr + k
        linear_vals = tl.load(linear_ptrs, mask=(tl.arange(0, k_end - k) < (k_end - k)), other=0.0)
        
        # Load y values for this block (handle broadcasting if needed)
        y_block = tl.load(y_base_ptr + k, mask=(tl.arange(0, k_end - k) < (k_end - k)), other=0.0)
        
        # Perform multiplication
        result = linear_vals * y_block.to(tl.float16).to(tl.float32)
        
        # Store final result
        final_ptrs = out_base_ptr + k
        tl.store(final_ptrs, result.to(tl.bfloat16), mask=(tl.arange(0, k_end - k) < (k_end - k)))

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_linear_multiply(w, x, y):
    """Fused linear transformation and multiplication"""
    batch_size, seq_len, in_features = x.shape
    out_features = w.shape[0]  # w is [in_features, out_features]
    
    # Output shape: [batch_size, seq_len, out_features]
    linear_out = torch.empty((batch_size, seq_len, out_features), dtype=torch.bfloat16, device=x.device)
    final_out = torch.empty((batch_size, seq_len, out_features), dtype=torch.bfloat16, device=x.device)
    
    # Triton kernel launch
    grid_size = batch_size * seq_len
    
    fused_linear_mul_kernel[grid_size](
        x_ptr=x,
        w_ptr=w.t(),  # Transpose weight for column-major access
        y_ptr=y,
        out_ptr=final_out,  # Store directly in final output
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        BLOCK_SIZE_M=1,  # Process one seq at a time
        BLOCK_SIZE_N=64,  # Output features per thread
        BLOCK_SIZE_K=128,  # Input features per thread
    )
    
    return linear_out, final_out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_linear_multiply