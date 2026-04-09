import torch
import triton
import triton.language as tl

# Pattern matching function - start with just the arithmetic operations
def pattern(in_0, in_1):
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for simple arithmetic fusion
@triton.jit
def fused_arithmetic_kernel(
    in_0_ptr,
    in_1_ptr, 
    out_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    feature_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program indices
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < (batch_size * seq_len * feature_dim)
    
    # Flatten indices for easier calculation
    flat_idx = pid
    
    # Load in_0 and convert to float32 for arithmetic
    in_0_offset = (flat_idx // (seq_len * feature_dim)) * seq_len + (flat_idx % seq_len) // feature_dim
    # Create mask for in_0 to ensure we don't access out of bounds
    in_0_mask = (in_0_offset >= 0) & (in_0_offset < batch_size * seq_len)
    in_0_val = tl.load(in_0_ptr + in_0_offset, mask=in_0_mask, other=0).to(tl.float32)
    in_0_scaled = in_0_val * 1000000.0
    
    # Load in_1
    in_1_val = tl.load(in_1_ptr + flat_idx, mask=mask, other=0.0)
    
    # Apply computation
    result = in_1_val - in_0_scaled
    
    # Store result
    tl.store(out_ptr + flat_idx, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_arithmetic(in_0, in_1):
    # Handle device placement - move in_0 to GPU if needed
    if in_0.device != in_1.device:
        in_0 = in_0.to(in_1.device)
    
    # Convert in_0 to the same dtype as in_1
    if in_0.dtype != in_1.dtype:
        in_0 = in_0.to(in_1.dtype)
    
    batch_size, seq_len, feature_dim = in_1.shape
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Determine block size and grid dimensions
    BLOCK_SIZE = 512
    total_elements = batch_size * seq_len * feature_dim
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_arithmetic_kernel[(num_programs,)](
        in_0,
        in_1,
        out,
        batch_size,
        seq_len, 
        feature_dim,
        BLOCK_SIZE
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_arithmetic