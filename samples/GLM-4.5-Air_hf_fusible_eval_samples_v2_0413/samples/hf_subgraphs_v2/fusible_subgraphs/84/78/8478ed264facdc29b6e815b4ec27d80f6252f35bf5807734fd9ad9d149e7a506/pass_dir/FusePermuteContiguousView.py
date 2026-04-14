import torch
import triton
import triton.language as tl

def pattern(x, target_shape):
    """
    Match the exact computation sequence:
    tmp_5 = x.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(target_shape)
    Returns tmp_7, tmp_5 to maintain observability
    """
    tmp_5 = x.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(target_shape)
    return tmp_7, tmp_5  # Return both results to maintain observability outside the pattern

def replacement_args(x, target_shape):
    return (x, target_shape)

@triton.jit
def permute_view_kernel(
    x_ptr, out_ptr, 
    batch_size, dim1, dim2, dim3,
    target_batch, target_dim1, target_dim2, target_dim3,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized kernel that performs permute(0,2,1,3) + view directly
    This avoids the intermediate contiguous operation
    """
    pid = tl.program_id(0)
    n_elements = target_batch * target_dim1 * target_dim2 * target_dim3
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate original coordinates from flat indices
    idx = offsets
    batch_idx = idx // (target_dim1 * target_dim2 * target_dim3)
    idx = idx % (target_dim1 * target_dim2 * target_dim3)
    dim1_idx = idx // (target_dim2 * target_dim3)
    idx = idx % (target_dim2 * target_dim3)
    dim2_idx = idx // target_dim3
    dim3_idx = idx % target_dim3
    
    # Map permuted(0,2,1,3) coordinates back to original coordinates
    # Original: [batch, dim1, dim2, dim3]
    # Permuted: [batch, dim2, dim1, dim3]
    orig_batch = batch_idx
    orig_dim1 = dim2_idx
    orig_dim2 = dim1_idx  
    orig_dim3 = dim3_idx
    
    # Compute original flat index
    orig_idx = ((orig_batch * dim1 + orig_dim1) * dim2 + orig_dim2) * dim3 + orig_dim3
    
    # Load data directly with bounds checking
    x = tl.load(x_ptr + orig_idx, mask=orig_idx < (batch_size * dim1 * dim2 * dim3), other=0.0)
    
    # Store result
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def optimized_permute_view(x, target_shape):
    """
    Optimized operation that combines permute(0,2,1,3) + contiguous + view
    """
    # Get input shape
    batch_size, dim1, dim2, dim3 = x.shape
    
    # Get target shape
    target_batch, target_dim1, target_dim2, target_dim3 = target_shape
    
    # Verify that total elements match
    assert batch_size * dim1 * dim2 * dim3 == target_batch * target_dim1 * target_dim2 * target_dim3, \
        "Target shape must have same number of elements as input"
    assert batch_size == target_batch, "Batch size must match"
    assert dim3 == target_dim3, "Last dimension must match"
    
    # Create output tensor
    result = torch.empty(target_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    n_elements = target_batch * target_dim1 * target_dim2 * target_dim3
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    permute_view_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=result,
        batch_size=batch_size,
        dim1=dim1, dim2=dim2, dim3=dim3,
        target_batch=target_batch,
        target_dim1=target_dim1, target_dim2=target_dim2, target_dim3=target_dim3,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result, x.permute(0, 2, 1, 3)  # Return permuted version for observability

def replacement_func():
    return optimized_permute_view