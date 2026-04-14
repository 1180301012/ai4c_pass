import torch
import triton
import triton.language as tl

# Pattern matching function for torch.cat with three tensors along dimension 2
def pattern(tensor_a, tensor_b, tensor_c):
    """Match torch.cat([tensor_a, tensor_b, tensor_c], 2) operation"""
    concat_result = torch.cat([tensor_a, tensor_b, tensor_c], 2)
    return concat_result

# Argument extraction function
def replacement_args(tensor_a, tensor_b, tensor_c):
    # Extract the three tensors to be concatenated
    return (tensor_a, tensor_b, tensor_c)

# Optimized Triton kernel for three-tensor concatenation along dimension 2
@triton.jit
def optimized_concat_3_tensors_kernel(
    ptr_a, ptr_b, ptr_c, output_ptr,
    size_a, size_b, size_c,  # sizes along dimension 2
    batch_size, channel_dim,  # sizes along other dimensions (assume same for all tensors)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for concatenating three tensors along dimension 2"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate total size along dimension 2
    total_size_dim2 = size_a + size_b + size_c
    
    # Compute indices for output tensor
    offset_flattened = offsets
    
    # Calculate position along dimension 2
    pos_dim2 = offset_flattened % total_size_dim2
    
    # Determine which input tensor this element comes from and its local offset
    if pos_dim2 < size_a:
        # From tensor_a
        local_pos_dim2 = pos_dim2
        src_ptr = ptr_a
    elif pos_dim2 < size_a + size_b:
        # From tensor_b  
        local_pos_dim2 = pos_dim2 - size_a
        src_ptr = ptr_b
    else:
        # From tensor_c
        local_pos_dim2 = pos_dim2 - size_a - size_b
        src_ptr = ptr_c
    
    # Convert flattened offset to multi-dimensional indexes
    # For shape [batch_size, 1, total_size_dim2], we have:
    batch_idx = offset_flattened // total_size_dim2
    batch_strided_offset = batch_idx * total_size_dim2
    
    # Calculate flattened index in source tensor
    src_flattened_offset = batch_strided_offset + local_pos_dim2
    
    # Load element from source tensor and store to output
    src_value = tl.load(src_ptr + src_flattened_offset, mask=mask, other=0.0)
    tl.store(output_ptr + offset_flattened, src_value, mask=mask)

# Kernel wrapper decorated with torch.fx.wrap  
@torch.fx.wrap
def optimized_concat_3_tensors(tensor_a, tensor_b, tensor_c):
    """Wrapper function for optimized 3-tensor concatenation along dimension 2"""
    # Verify input tensors have compatible shapes
    assert tensor_a.shape[0] == tensor_b.shape[0] == tensor_c.shape[0], "Batch sizes must match"
    assert tensor_a.shape[1] == tensor_b.shape[1] == tensor_c.shape[1], "Channel dimensions must match"
    
    batch_size, channel_dim = tensor_a.shape[0], tensor_a.shape[1]
    size_a = tensor_a.shape[2]
    size_b = tensor_b.shape[2] 
    size_c = tensor_c.shape[2]
    
    # Calculate output tensor shape and total elements
    output_size_dim2 = size_a + size_b + size_c
    n_elements = batch_size * channel_dim * output_size_dim2
    
    BLOCK_SIZE = 1024  # Optimal block size for concatenation
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    output_shape = [batch_size, channel_dim, output_size_dim2]
    output = torch.empty(output_shape, dtype=tensor_a.dtype, device=tensor_a.device)
    
    # Launch the optimized concatenation kernel
    optimized_concat_3_tensors_kernel[(num_programs,)](
        ptr_a=tensor_a,
        ptr_b=tensor_b, 
        ptr_c=tensor_c,
        output_ptr=output,
        size_a=size_a,
        size_b=size_b,
        size_c=size_c,
        batch_size=batch_size,
        channel_dim=channel_dim,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function - returns the kernel wrapper (not called)
def replacement_func():
    return optimized_concat_3_tensors