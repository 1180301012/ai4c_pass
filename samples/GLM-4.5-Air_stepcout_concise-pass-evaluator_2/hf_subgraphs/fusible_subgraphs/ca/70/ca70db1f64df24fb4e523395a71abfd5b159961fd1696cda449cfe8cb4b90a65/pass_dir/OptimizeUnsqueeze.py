import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Simple unsqueeze operation
    tmp_13 = input_tensor.unsqueeze(-2)
    
    # Return the expanded tensor
    return tmp_13

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def unsqueeze_kernel(
    input_ptr,
    output_ptr,
    n_batch, n_in,
    new_pos: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE_M * BLOCK_SIZE_N
    offsets_m = block_start // (BLOCK_SIZE_M * BLOCK_SIZE_N)
    offsets_n = block_start % (BLOCK_SIZE_M * BLOCK_SIZE_N)
    
    # Create 2D grid within the block
    local_m = tl.arange(0, BLOCK_SIZE_M)
    local_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Add base offsets
    global_m = offsets_m + local_m[:, None]
    global_n = (offsets_n + local_n) % n_in
    
    # Mask for valid positions
    mask_m = global_m < n_batch
    mask_n = global_n < n_in
    
    # Load input
    input_vals = tl.load(input_ptr + global_m * n_in + global_n,
                        mask=mask_m & mask_n, other=0.0)
    
    # Output is [n_batch, 1, n_in], so we need to handle the new dimension
    # Create output positions with the new dimension
    output_m = global_m
    output_n = global_n
    
    # Store to output [n_batch, 1, n_in]
    # The new dimension is fixed at position 1 with size 1
    output_ptr_flat = output_ptr + (output_m * 1 * n_in + 0 * n_in + output_n)
    tl.store(output_ptr_flat, input_vals, mask=mask_m & mask_n)

@torch.fx.wrap
def optimized_unsqueeze(input_tensor):
    # Input tensor shape: [batch_size, input_dim]
    n_batch, n_in = input_tensor.shape
    
    # Output tensor shape: [batch_size, 1, input_dim]
    output_shape = (n_batch, 1, n_in)
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Triton kernel launch parameters
    BLOCK_SIZE_M = 64  # Batch dimension block size
    BLOCK_SIZE_N = 256  # Input dimension block size
    
    grid = lambda meta: (
        (n_batch * n_in + BLOCK_SIZE_M * BLOCK_SIZE_N - 1) // (BLOCK_SIZE_M * BLOCK_SIZE_N),
    )
    
    unsqueeze_kernel[grid](
        input_tensor,
        output_tensor,
        n_batch, n_in,
        -2,  # New dimension position
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return output_tensor

def replacement_func():
    return optimized_unsqueeze