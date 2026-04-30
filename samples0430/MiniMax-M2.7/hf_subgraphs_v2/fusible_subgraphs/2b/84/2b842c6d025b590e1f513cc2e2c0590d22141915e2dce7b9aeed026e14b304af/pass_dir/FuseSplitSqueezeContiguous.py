import torch
import triton
import triton.language as tl

# Pattern to match the multiply and subtract operations
# This is the core computation that can be optimized
def pattern(in_0, in_1):
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    return tmp_2

# Extract arguments from matched nodes
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Autotune configuration for optimal performance
autotune_configs = [
    triton.Config({"BLOCK_SIZE": 32}, num_stages=1, num_warps=1),
    triton.Config({"BLOCK_SIZE": 64}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_SIZE": 128}, num_stages=2, num_warps=4),
]

@triton.autotune(
    configs=autotune_configs,
    key=["n_elements"],
)
@triton.jit
def fused_mul_sub_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    batch_size,
    seq_len,
    last_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global position
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute position indices
    batch_idx = offsets // (seq_len * last_dim)
    remainder = offsets % (seq_len * last_dim)
    seq_idx = remainder // last_dim
    last_idx = remainder % last_dim
    
    # Load in_0 (shape: [batch, seq, 1]) - broadcast to [batch, seq, last_dim]
    in_0_offsets = batch_idx * seq_len + seq_idx
    in_0 = tl.load(in_0_ptr + in_0_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Multiply by 1000000.0
    scaled_val = in_0 * 1000000.0
    
    # Load in_1 element
    in_1_offsets = batch_idx * seq_len * last_dim + seq_idx * last_dim + last_idx
    in_1 = tl.load(in_1_ptr + in_1_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute result
    result = in_1 - scaled_val
    
    # Store result
    out_offsets = batch_idx * seq_len * last_dim + seq_idx * last_dim + last_idx
    tl.store(out_ptr + out_offsets, result, mask=mask)


@torch.fx.wrap
def fused_mul_sub_wrapper(in_0, in_1):
    """Wrapper for the fused multiply-subtract kernel"""
    # Get shape info
    batch_size = in_1.shape[0]
    seq_len = in_1.shape[1]
    last_dim = in_1.shape[2] if len(in_1.shape) > 2 else 1
    n_elements = batch_size * seq_len * last_dim
    
    # Allocate output tensor
    out = torch.empty_like(in_1)
    
    # Grid size - use 1D grid for simplicity
    grid = ((n_elements + 128 - 1) // 128,)
    
    # Call the autotuned kernel
    fused_mul_sub_kernel[grid](
        in_0,
        in_1,
        out,
        n_elements,
        batch_size,
        seq_len,
        last_dim,
    )
    
    return out


def replacement_func():
    return fused_mul_sub_wrapper