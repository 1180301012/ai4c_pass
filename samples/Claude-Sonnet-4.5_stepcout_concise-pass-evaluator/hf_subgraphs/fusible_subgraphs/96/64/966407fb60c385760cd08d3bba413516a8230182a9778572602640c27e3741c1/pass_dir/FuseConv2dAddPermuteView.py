import torch
import triton
import triton.language as tl


def pattern(input_tensor):
    """
    Pattern matches: permute + contiguous
    """
    result = input_tensor.permute(0, 2, 1, 3)
    result = result.contiguous()
    return result


def replacement_args(input_tensor):
    return (input_tensor,)


@triton.jit
def fused_permute_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    dim1,  # 4 (groups)
    dim2,  # H (sequence length)
    dim3,  # 8 (head dim)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: input.permute(0, 2, 1, 3).contiguous()
    Input shape: [batch_size, dim1, dim2, dim3]
    Output shape: [batch_size, dim2, dim1, dim3]
    """
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Total elements in output
    n_elements = batch_size * dim2 * dim1 * dim3
    mask = offsets < n_elements
    
    # Calculate output indices
    b_idx = offsets // (dim2 * dim1 * dim3)
    remainder = offsets % (dim2 * dim1 * dim3)
    d2_idx = remainder // (dim1 * dim3)
    remainder = remainder % (dim1 * dim3)
    d1_idx = remainder // dim3
    d3_idx = remainder % dim3
    
    # Calculate input indices (before permute)
    # Input layout: [batch, dim1, dim2, dim3]
    in_offsets = (b_idx * dim1 * dim2 * dim3 + 
                  d1_idx * dim2 * dim3 + 
                  d2_idx * dim3 + 
                  d3_idx)
    
    # Load 
    in_val = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    
    # Store to permuted location
    tl.store(out_ptr + offsets, in_val, mask=mask)


@torch.fx.wrap
def fused_permute_view(input_tensor):
    """
    Optimized implementation:
    Fuse permute + contiguous into single kernel
    """
    # Get shapes
    batch_size, dim1, dim2, dim3 = input_tensor.shape
    
    # Allocate output with permuted shape
    out = torch.empty((batch_size, dim2, dim1, dim3), 
                      dtype=input_tensor.dtype, 
                      device=input_tensor.device)
    
    # Launch fused kernel
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_permute_kernel[grid](
        input_tensor,
        out,
        batch_size,
        dim1,
        dim2,
        dim3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_permute_view