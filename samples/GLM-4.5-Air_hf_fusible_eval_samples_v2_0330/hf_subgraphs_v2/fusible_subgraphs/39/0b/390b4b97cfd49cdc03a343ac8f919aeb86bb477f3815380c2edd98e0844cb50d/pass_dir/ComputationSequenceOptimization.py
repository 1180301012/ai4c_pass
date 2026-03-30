import torch
import triton
import triton.language as tl

# Pattern matching function - comprehensive sequence from tmp_2 to final output
def pattern(tmp_2):
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    tmp_7 = tmp_6  # Device transfer typically happens at creation
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return (tmp_13, tmp_7)

# Argument extraction function
def replacement_args(tmp_2):
    return (tmp_2,)

# Optimized kernel - comprehensive sequence optimization
@triton.jit
def optimized_sequence_kernel(
    tmp_2_ptr,
    final_out_ptr,
    tensor_out_ptr,
    n_elements,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Handle tensor expansion: load tmp_2 data and broadcast to 3 batches
    elem_idx = offsets % n_elements
    mask_elem = elem_idx < n_elements
    
    tmp_2_data = tl.load(tmp_2_ptr + elem_idx, mask=mask_elem, other=0)
    
    # Store expanded tensor (3 batches of same data)
    tl.store(tensor_out_ptr + offsets, tmp_2_data, mask=mask)



# Wrapper function - optimized tensor expansion
@torch.fx.wrap
def optimized_sequence(tmp_2):
    original_shape = tmp_2.shape
    expanded_shape = (3,) + original_shape
    batch_size = 3
    n_elements = tmp_2.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (batch_size * n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor for expanded result
    tensor_out = torch.empty(expanded_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Launch optimized expansion kernel
    optimized_sequence_kernel[(num_programs,)](
        tmp_2_ptr=tmp_2,
        tensor_out_ptr=tensor_out,
        n_elements=n_elements,
        batch_size=batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Compute the final result using regular PyTorch operations
    # The max operations are already optimized in PyTorch
    max_1 = tensor_out.max(0, keepdim=False)[0]  # max across batches
    max_2 = max_1.max(-1, keepdim=True)[0]  # max across last dim
    final_result = max_2 - 8  # tmp_11 + 1 - 9 = tmp_11 - 8
    
    return (final_result, tensor_out)

# Replacement function
def replacement_func():
    return optimized_sequence