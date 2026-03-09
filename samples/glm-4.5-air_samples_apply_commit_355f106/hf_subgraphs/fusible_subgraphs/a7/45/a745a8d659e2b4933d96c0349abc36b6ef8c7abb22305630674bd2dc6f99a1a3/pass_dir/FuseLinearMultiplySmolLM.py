import torch
import triton
import triton.language as tl

# Pattern matching function for SmolLM3-3B: linear + multiply pattern
def pattern(weight, hidden_states, silu_output):
    """
    Pattern that matches: 
    tmp_0 = weight
    tmp_1 = hidden_states
    tmp_2 = silu_output
    tmp_3 = linear_transformation(tmp_1, tmp_0)  # Will be replaced
    tmp_4 = tmp_2 * tmp_3
    return (tmp_4,)
    """
    # Create temporary variables matching the original computation
    tmp_0 = weight
    tmp_1 = hidden_states
    tmp_2 = silu_output
    
    # This will be replaced by the fused kernel
    tmp_3 = tmp_1 @ tmp_0  # Placeholder - will be replaced
    tmp_4 = tmp_2 * tmp_3
    
    return (tmp_4,)

# Argument extraction function
def replacement_args(weight, hidden_states, silu_output):
    return (weight, hidden_states, silu_output)

# Optimized Triton kernel for fused linear + multiply
@triton.jit
def fused_linear_multiply_kernel_smolllm(
    weight_ptr,           # Weight matrix [out_features, in_features]
    hidden_states_ptr,    # Hidden states [batch, seq_len, in_features]
    silu_output_ptr,      # SiLU output [batch, seq_len, out_features]
    output_ptr,           # Output tensor [batch, seq_len, out_features]
    batch_size,
    seq_len,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program id for batch and sequence dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Ensure we don't go out of bounds
    if pid_m >= batch_size or pid_n >= seq_len:
        return
    
    # Compute base offset for this batch and sequence position
    hs_offset = (pid_m * seq_len + pid_n) * in_features
    silu_offset = (pid_m * seq_len + pid_n) * out_features
    
    # Load hidden states data
    hidden_states = tl.load(hidden_states_ptr + hs_offset + tl.arange(0, BLOCK_SIZE_N),
                           mask=tl.arange(0, BLOCK_SIZE_N) < in_features,
                           other=0.0)
    
    # Load SiLU output data
    silu_output = tl.load(silu_output_ptr + silu_offset + tl.arange(0, BLOCK_SIZE_N),
                         mask=tl.arange(0, BLOCK_SIZE_N) < out_features,
                         other=0.0)
    
    # Matrix multiplication for linear transformation with element-wise multiply fused
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Load weight matrix chunk
        weight_chunk = tl.load(weight_ptr + 
                             k * out_features + 
                             tl.arange(0, BLOCK_SIZE_N),
                             mask=tl.arange(0, BLOCK_SIZE_N) < out_features,
                             other=0.0)
        
        # Load hidden states chunk
        hs_chunk = tl.load(hidden_states_ptr + 
                          hs_offset + k + 
                          tl.arange(0, BLOCK_SIZE_K),
                          mask=tl.arange(0, BLOCK_SIZE_K) < (in_features - k),
                          other=0.0)
        
        # Matrix multiplication accumulator
        acc += weight_chunk * hs_chunk[:, None]
    
    # Apply element-wise multiplication with SiLU output
    linear_result = acc.to(tl.float32)
    fused_result = silu_output * linear_result
    
    # Store final result
    tl.store(output_ptr + silu_offset + tl.arange(0, BLOCK_SIZE_N),
             fused_result, mask=tl.arange(0, BLOCK_SIZE_N) < out_features)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_linear_multiply_smolllm(weight, hidden_states, silu_output):
    # Get tensor shapes
    batch_size = hidden_states.size(0)
    seq_len = hidden_states.size(1)
    in_features = hidden_states.size(2)
    out_features = weight.size(0)
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, out_features), 
                        dtype=hidden_states.dtype, device=hidden_states.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_M),
            triton.cdiv(seq_len, BLOCK_SIZE_N))
    
    fused_linear_multiply_kernel_smolllm[grid](
        weight,
        hidden_states,
        silu_output,
        output,
        batch_size,
        seq_len,
        in_features,
        out_features,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return (output,)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_linear_multiply_smolllm