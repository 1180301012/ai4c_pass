import torch
import triton
import triton.language as tl

def pattern(in_2):
    """Pattern for fused RMS normalization operations"""
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    return tmp_16

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def rms_norm_kernel(
    input_ptr,
    output_ptr,
    input_size,
    hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused RMS normalization kernel"""
    # Get program ID and create offset trackers
    pid = tl.program_id(0)
    m_offset = pid * BLOCK_SIZE_M
    
    # Initialize accumulator for squared sum
    squared_sum = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    # Load and compute squared values across sequence dimension
    for _ in range(0, input_size, BLOCK_SIZE_N):
        n_offset = tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offset < (input_size - m_offset)
        
        # Load input data (convert to float32 internally)
        input_val = tl.load(input_ptr + (m_offset * input_size + n_offset), 
                           mask=n_mask, other=0.0).to(tl.float32)
        
        # Accumulate squared values
        squared_sum += input_val * input_val
    
    # Compute mean across hidden dimension (already averaged over sequence)
    mean = squared_sum / input_size
    
    # Add epsilon and compute reciprocal square root
    inv_std = tl.rsqrt(mean + eps)
    
    # Normalize and convert back to original dtype
    for _ in range(0, input_size, BLOCK_SIZE_N):
        n_offset = tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offset < (input_size - m_offset)
        
        # Load input data
        input_val = tl.load(input_ptr + (m_offset * input_size + n_offset), 
                           mask=n_mask, other=0.0).to(tl.float32)
        
        # Apply normalization
        normalized = input_val * inv_std
        
        # Convert to bfloat16 and store
        output_val = normalized.to(tl.bfloat16)
        tl.store(output_ptr + (m_offset * input_size + n_offset), output_val, mask=n_mask)

@torch.fx.wrap
def fused_rms_norm(in_2):
    """Wrapper function for fused RMS normalization"""
    # Get input shape information
    input_shape = in_2.shape
    if len(input_shape) == 3:
        batch_size, seq_len, hidden_size = input_shape
        input_size = seq_len * hidden_size
    else:
        # Fallback for other shapes
        batch_size, seq_len, hidden_size = 1, input_shape[-2], input_shape[-1]
        input_size = seq_len * hidden_size
    
    # Create output tensor
    output = torch.empty_like(in_2, dtype=torch.bfloat16)
    
    # Set optimal block sizes based on typical transformer dimensions
    BLOCK_SIZE_M = 64   # Process multiple tokens simultaneously
    BLOCK_SIZE_N = 256  # Hidden dimension block size
    
    # Calculate grid size
    num_programs = (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    
    # Launch kernel
    rms_norm_kernel[(num_programs,)](
        input_ptr=in_2,
        output_ptr=output,
        input_size=input_size,
        hidden_size=hidden_size,
        eps=1e-06,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return fused_rms_norm