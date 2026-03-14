import torch
import triton
import triton.language as tl

# Pattern matching function for transformers: SwiGLU-style linear + element-wise multiplication
def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = torch.nn.functional.linear(in_1, tmp_0, None)
    tmp_2 = in_2 * tmp_1
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Triton kernel for optimized SwiGLU operation (linear + multiplication fusion)
@triton.jit
def swiglu_kernel(
    weight_ptr,
    hidden_states_ptr,
    silu_output_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_size,
    intermediate_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Compute program IDs and offsets
    pid = tl.program_id(0)
    
    # Calculate batch and sequence position
    batch_seq = pid * BLOCK_SIZE_M
    batch_id = batch_seq // (seq_len * hidden_size)
    seq_id = (batch_seq % (seq_len * hidden_size)) // hidden_size
    offset_in_seq = batch_seq % hidden_size
    
    # Ensure we don't go out of bounds
    if batch_id >= batch_size or seq_id >= seq_len:
        return
    
    # Calculate memory offsets
    hidden_offset = batch_id * seq_len * hidden_size + seq_id * hidden_size
    intermediate_offset = batch_id * seq_len * intermediate_size + seq_id * intermediate_size
    
    # Load hidden states slice
    hidden_offsets = hidden_offset + tl.arange(0, BLOCK_SIZE_K)
    hidden_mask = hidden_offsets < (batch_id * seq_len + seq_id + 1) * hidden_size
    hidden_slice = tl.load(hidden_states_ptr + hidden_offsets, mask=hidden_mask, other=0.0)
    
    # Load weight matrix slice
    weight_offsets = tl.arange(0, BLOCK_SIZE_N * BLOCK_SIZE_K)
    weight_mask = weight_offsets < (intermediate_size * hidden_size)
    weight_slice = tl.load(weight_ptr + weight_offsets, mask=weight_mask)
    
    # Compute linear part: y = hidden_states @ weight.T
    linear_result = tl.dot(hidden_slice, weight_slice)
    
    # Load SiLU output slice
    silu_offsets = intermediate_offset + tl.arange(0, BLOCK_SIZE_N)
    silu_mask = silu_offsets < (batch_id * seq_len + seq_id + 1) * intermediate_size
    silu_slice = tl.load(silu_output_ptr + silu_offsets, mask=silu_mask, other=0.0)
    
    # Compute SwiGLU: output = silu * linear_result
    final_result = silu_slice * linear_result
    
    # Store result
    tl.store(output_ptr + silu_offsets, final_result, mask=silu_mask)

# Optimized kernel wrapper
@torch.fx.wrap
def optimized_swiglu_mul(in_0, in_1, in_2):
    # Get tensor shapes
    batch_size = in_1.size(0)
    seq_len = in_1.size(1) if in_1.dim() > 2 else 1
    hidden_size = in_1.size(-1)
    intermediate_size = in_0.size(0)
    
    # Create output tensor
    output = torch.empty_like(in_2)
    
    # Tile size configuration - optimized for typical transformer sizes
    BLOCK_SIZE_M = 1  # Process one element at a time for simplicity
    BLOCK_SIZE_N = min(256, intermediate_size)
    BLOCK_SIZE_K = min(128, hidden_size)
    
    # Calculate grid size
    total_elements = batch_size * seq_len * intermediate_size
    grid_size = (total_elements + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    swiglu_kernel[grid_size](
        in_0,
        in_1,
        in_2,
        output,
        batch_size,
        seq_len,
        hidden_size,
        intermediate_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_swiglu_mul