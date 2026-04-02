import torch
import triton
import triton.language as tl

def pattern(weight, input_tensor):
    # Simplified pattern: just match the linear projection
    # Let's start with a basic pattern that should match any linear op
    tmp_1 = torch.nn.functional.linear(input_tensor, weight, None)
    return tmp_1

def replacement_args(weight, input_tensor):
    return (weight, input_tensor)

@triton.jit
def linear_fused_kernel(
    output_ptr,           # [batch, seq_len, features_out] 
    input_ptr,            # [batch, seq_len, features_in] 
    weight_ptr,           # [features_in, features_out]
    batch,                # batch size (always 1)
    seq_len,              # sequence length (always 197)  
    features_in,          # input features
    features_out,         # output features (3 * reshape_N * 48)
    BLOCK_SIZE_M: tl.constexpr,  # Block size for M dimension (batch * seq_len)
    BLOCK_SIZE_N: tl.constexpr,  # Block size for N dimension (features_out)
    BLOCK_SIZE_K: tl.constexpr,  # Block size for K dimension (features_in)
):
    # Each program handles a BLOCK_SIZE_M x BLOCK_SIZE_N tile
    pid_m = tl.program_id(0)  # M dimension (batch * seq_len)
    pid_n = tl.program_id(1)  # N dimension (features_out)
    
    # Compute coordinates
    m_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks for bounds checking
    m_mask = m_idx < batch * seq_len
    n_mask = n_idx < features_out
    
    # Split m_idx into batch and sequence
    b_idx = m_idx // seq_len
    s_idx = m_idx % seq_len
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Simplified K loop
    for k in range(0, features_in, BLOCK_SIZE_K):
        # Compute K indices with bounds checking
        k_mask = k + tl.arange(0, BLOCK_SIZE_K) < features_in
        
        # Load input: [batch, seq_len, features_in]
        input_addr = (b_idx[:, None] * seq_len + s_idx[:, None]) * features_in + (k + tl.arange(0, BLOCK_SIZE_K))[None, :]
        input_vals = tl.load(
            input_ptr + input_addr,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Load weights: [features_in, features_out] 
        weight_addr = (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * features_out + n_idx[None, :]
        weight_vals = tl.load(
            weight_ptr + weight_addr,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        )
        
        # Matrix multiply
        acc += tl.dot(input_vals, weight_vals)
    
    # Store result
    output_addr = b_idx[:, None] * seq_len * features_out + \
                  s_idx[:, None] * features_out + \
                  n_idx[None, :]
    
    tl.store(
        output_ptr + output_addr,
        acc,
        mask=m_mask[:, None] & n_mask[None, :]
    )

@torch.fx.wrap  
def optimized_linear(weight, input_tensor):
    # Optimized linear operation with smaller block sizes for reduced overhead
    batch, seq_len, features_in = input_tensor.shape
    features_out = weight.shape[0]
    
    # Create output tensor
    output = torch.empty((batch, seq_len, features_out), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # Use smaller block sizes to reduce overhead for these smaller operations
    BLOCK_SIZE_M = 32     # Each program handles fewer sequence positions  
    BLOCK_SIZE_N = 64     # Smaller blocks for better cache utilization
    BLOCK_SIZE_K = 16     # Smaller K block to improve cache locality
    
    # Calculate grid size
    grid_m = (batch * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (features_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    linear_fused_kernel[grid_m, grid_n](
        output_ptr=output,
        input_ptr=input_tensor,
        weight_ptr=weight,
        batch=batch,
        seq_len=seq_len,
        features_in=features_in,
        features_out=features_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return optimized_linear