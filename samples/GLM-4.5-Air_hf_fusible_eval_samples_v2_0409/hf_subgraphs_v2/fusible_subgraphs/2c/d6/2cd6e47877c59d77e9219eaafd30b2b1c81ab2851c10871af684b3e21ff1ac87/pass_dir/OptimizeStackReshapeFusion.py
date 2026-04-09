import torch
import triton
import triton.language as tl

def pattern(tensor1, tensor2, target_shape):
    """Pattern matching for stack + reshape fusion"""
    # Stack the tensors along the last dimension
    stacked = torch.stack([tensor1, tensor2], -1)
    
    # Reshape to target shape - stacked becomes dead after this
    reshaped = stacked.reshape(target_shape)
    
    # Only return the reshaped result since stacked becomes dead code
    return reshaped

def replacement_args(tensor1, tensor2, target_shape):
    return (tensor1, tensor2, target_shape)

@triton.jit
def fused_stack_reshape_kernel(
    tensor1_ptr,
    tensor2_ptr,
    output_ptr,
    target_shape_0,
    target_shape_1,
    target_shape_2,
    target_shape_3,
    n_elements_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate element indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_total
    
    # Load elements from both tensors
    # Since we're stacking along dimension -1, we need interleave the data
    total_elements = n_elements_total
    
    # Calculate how many elements to process per tensor in this block
    elements_per_tensor = total_elements // 2
    
    # Load tensor1 elements (even indices in output)
    tensor1_idx = offsets // 2
    tensor1_idx_tensor = tl.tensor.make_contiguous(tensor1_idx)
    tensor1_mask = tensor1_idx_tensor < elements_per_tensor
    
    if tl.constexpr(elements_per_tensor > 0):
        tensor1_data = tl.load(tensor1_ptr + tensor1_idx_tensor, mask=tensor1_mask, other=0.0)
    else:
        tensor1_data = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    
    # Load tensor2 elements (odd indices in output)  
    tensor2_idx = offsets // 2
    tensor2_idx_tensor = tl.tensor.make_contiguous(tensor2_idx)
    tensor2_mask = tensor2_idx_tensor < elements_per_tensor
    
    if tl.constexpr(elements_per_tensor > 0):
        tensor2_data = tl.load(tensor2_ptr + tensor2_idx_tensor, mask=tensor2_mask, other=0.0)
    else:
        tensor2_data = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
    
    # Fuse stack and reshape: interleave the data
    # For each position in output, we alternate between tensor1 and tensor2 data
    result_data = tl.where(
        (offsets % 2) == 0,
        tl.tensor.make_contiguous(tensor1_data),
        tl.tensor.make_contiguous(tensor2_data)
    )
    
    # Store the result
    tl.store(output_ptr + offsets, result_data, mask=mask)

@torch.fx.wrap
def optimized_stack_reshape(tensor1, tensor2, target_shape):
    """Optimized stack + reshape operation"""
    # Create output tensor with target shape
    output = torch.empty(target_shape, dtype=tensor1.dtype, device=tensor1.device)
    
    # Calculate total elements
    n_elements_total = output.numel()
    
    # Block size for kernel launch
    BLOCK_SIZE = 1024
    num_programs = (n_elements_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_stack_reshape_kernel[(num_programs,)](
        tensor1_ptr=tensor1,
        tensor2_ptr=tensor2,
        output_ptr=output,
        target_shape_0=target_shape[0] if len(target_shape) > 0 else 1,
        target_shape_1=target_shape[1] if len(target_shape) > 1 else 1,
        target_shape_2=target_shape[2] if len(target_shape) > 2 else 1,
        target_shape_3=target_shape[3] if len(target_shape) > 3 else 1,
        n_elements_total=n_elements_total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, tensor1, tensor2

@torch.fx.wrap
def optimized_stack_reshape(tensor1, tensor2, target_shape):
    """Optimized stack + reshape operation"""
    # Create output tensor with target shape
    output = torch.empty(target_shape, dtype=tensor1.dtype, device=tensor1.device)
    
    # Calculate total elements
    n_elements_total = output.numel()
    
    # Block size for kernel launch
    BLOCK_SIZE = 1024
    num_programs = (n_elements_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_stack_reshape_kernel[(num_programs,)](
        tensor1_ptr=tensor1,
        tensor2_ptr=tensor2,
        output_ptr=output,
        target_shape_0=target_shape[0] if len(target_shape) > 0 else 1,
        target_shape_1=target_shape[1] if len(target_shape) > 1 else 1,
        target_shape_2=target_shape[2] if len(target_shape) > 2 else 1,
        target_shape_3=target_shape[3] if len(target_shape) > 3 else 1,
        n_elements_total=n_elements_total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_stack_reshape