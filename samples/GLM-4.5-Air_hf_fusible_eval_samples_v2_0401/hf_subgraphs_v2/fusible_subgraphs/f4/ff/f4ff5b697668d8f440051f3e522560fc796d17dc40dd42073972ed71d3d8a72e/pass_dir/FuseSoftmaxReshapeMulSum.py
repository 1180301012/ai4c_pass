import torch
import triton
import triton.language as tl

def pattern(softmax_out, in_0, in_1):
    """
    Pattern to match: softmax + reshape + multiply + reshape + sum operations
    This pattern matches the core computation found in all the graph variants
    """
    # Reshape softmax output to 4D for effective spatial processing
    reshaped = softmax_out.reshape(-1, 17, 64, 64)
    
    # Multiply with in_0 and reshape back for reduction
    mul0 = reshaped.mul(in_0)
    reshaped_mul0 = mul0.reshape(softmax_out.shape[0], 17, -1)
    sum0 = torch.sum(reshaped_mul0, dim=2, keepdim=True)
    
    # Multiply with in_1 and reshape back for reduction  
    mul1 = reshaped.mul(in_1)
    reshaped_mul1 = mul1.reshape(softmax_out.shape[0], 17, -1)
    sum1 = torch.sum(reshaped_mul1, dim=2, keepdim=True)
    
    # Concatenate the results
    concatenated = torch.cat([sum0, sum1], dim=-1)
    
    return reshaped, concatenated

def replacement_args(softmax_out, in_0, in_1):
    return (softmax_out, in_0, in_1)

@triton.jit
def fused_softmax_mul_sum_kernel(
    softmax_ptr,
    in_0_ptr, 
    in_1_ptr,
    out_reshaped_ptr,
    out_sum_ptr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    spatial_size: tl.constexpr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized kernel that fuses softmax + reshape + multiply + sum operations
    """
    pid = tl.program_id(0)
    
    # Each program processes one element in the batch x heads space
    if pid >= batch_size * num_heads:
        return
        
    # Load softmax output for this batch,head combination
    softmax_offset = pid * spatial_size
    softmax_vals = tl.load(softmax_ptr + softmax_offset, mask=softmax_offset < total_elements)
    
    # Load in_0 and in_1 values
    in_0_val = tl.load(in_0_ptr)
    in_1_val = tl.load(in_1_ptr)
    
    # Compute spatial processing loop
    spatial_offset = pid * spatial_size
    if spatial_offset < total_elements:
        # Write reshaped output (1D flattening of spatial dimensions)
        tl.store(out_reshaped_ptr + spatial_offset, softmax_vals)
        
        # Compute sums for both weight tensors
        sum0 = tl.sum(softmax_vals * in_0_val)
        sum1 = tl.sum(softmax_vals * in_1_val)
        
        # Write sum results
        sum_offset = pid * 2
        tl.store(out_sum_ptr + sum_offset, sum0)
        tl.store(out_sum_ptr + sum_offset + 1, sum1)

@torch.fx.wrap
def fused_softmax_mul_sum(softmax_out, in_0, in_1):
    """
    Wrapper function for the fused computation
    """
    # Get tensor shapes and properties
    batch_size = softmax_out.shape[0]
    num_heads = softmax_out.shape[1] 
    spatial_size = softmax_out.shape[2]
    total_elements = softmax_out.numel()
    
    # Create output tensors
    out_reshaped = torch.empty_like(softmax_out)
    out_sum = torch.empty((batch_size, num_heads, 2), dtype=softmax_out.dtype, device=softmax_out.device)
    
    # Optimize block size based on tensor sizes
    if spatial_size <= 1024:
        BLOCK_SIZE = 1024
    elif spatial_size <= 2048:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    
    # Launch kernel
    grid_size = (batch_size * num_heads + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_softmax_mul_sum_kernel[grid_size](
        softmax_out,
        in_0,
        in_1, 
        out_reshaped,
        out_sum,
        batch_size,
        num_heads,
        spatial_size,
        total_elements,
        BLOCK_SIZE
    )
    
    return out_reshaped, out_sum

def replacement_func():
    """
    Returns the fused function as a zero-argument callable
    """
    return fused_softmax_mul_sum