import torch
import triton
import triton.language as tl

def pattern(tmp_11):
    tmp_12 = torch.nn.functional.softmax(tmp_11, dim=-1)
    tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
    return tmp_13

def replacement_args(tmp_11):
    return (tmp_11,)

@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a row of the softmax
    row_id = tl.program_id(0)
    col_start = row_id * dim_size
    
    # Load the row
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_id + 1) * dim_size
    x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x, mask=mask)
    tl.store(out_ptr + row_id, max_val, mask=True)
    
    # Compute exponentials
    x_exp = tl.exp(x - max_val)
    
    # Compute sum for normalization
    sum_exp = tl.sum(x_exp, mask=mask)
    
    # Store exponentials (we'll normalize later)
    tl.store(x_ptr + offsets, x_exp, mask=mask)

@triton.jit
def normalize_kernel(
    x_ptr,
    out_ptr,
    max_vals_ptr,
    n_elements,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a row
    row_id = tl.program_id(0)
    col_start = row_id * dim_size
    max_val = tl.load(max_vals_ptr + row_id)
    
    # Load exponentials
    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_id + 1) * dim_size
    x_exp = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize
    softmax_out = x_exp / tl.sum(x_exp, mask=mask)
    
    # Store result
    tl.store(out_ptr + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def triton_softmax_dropout(tmp_11):
    """
    Custom Triton implementation of softmax with dropout (p=0.0).
    Since dropout with p=0.0 is a no-op, we just implement softmax.
    """
    n_elements = tmp_11.numel()
    dim_size = tmp_11.shape[-1]  # Last dimension for softmax
    
    # Create output tensors
    softmax_out = torch.empty_like(tmp_11)
    max_vals = torch.empty(tmp_11.shape[:-1], device=tmp_11.device, dtype=tmp_11.dtype)
    
    # Launch kernels
    grid = (tmp_11.shape[:-1].numel(),)
    
    # First pass: compute row-wise maximums
    softmax_kernel[grid](
        tmp_11,
        max_vals,
        n_elements,
        dim_size,
        BLOCK_SIZE=min(1024, dim_size)
    )
    
    # Second pass: compute exponentials and normalize
    normalize_kernel[grid](
        tmp_11,
        softmax_out,
        max_vals,
        n_elements,
        dim_size,
        BLOCK_SIZE=min(1024, dim_size)
    )
    
    return softmax_out

def replacement_func():
    return triton_softmax_dropout