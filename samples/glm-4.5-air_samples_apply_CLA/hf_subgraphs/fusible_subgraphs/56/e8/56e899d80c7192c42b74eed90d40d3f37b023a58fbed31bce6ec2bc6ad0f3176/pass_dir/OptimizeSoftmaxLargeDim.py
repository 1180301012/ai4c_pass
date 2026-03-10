import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_3 = x
    tmp_4 = tmp_3.softmax(dim=-1)
    tmp_3 = None
    return tmp_4

def replacement_args(x):
    return (x,)

@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    
    # Compute exp(x - max)
    exp_x = tl.exp(x - max_val)
    
    # Compute sum
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Normalize
    out = exp_x / sum_exp
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_softmax(x):
    if x.dim() == 1:
        # For 1D case
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out = torch.empty_like(x)
        softmax_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
    else:
        # Handle multi-dimensional case by flattening the target dimension
        orig_shape = x.shape
        if len(orig_shape) >= 2:
            # Keep all dimensions except the last one
            flattened_dims = 1
            for dim in orig_shape[:-1]:
                flattened_dims *= dim
            target_dim = orig_shape[-1]
            
            # Reshape to [flattened_dims, target_dim]
            x_reshaped = x.reshape(flattened_dims, target_dim)
            
            # Apply softmax to each row
            BLOCK_SIZE = 1024
            cols = target_dim
            rows = flattened_dims
            num_programs = (rows * cols + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            out_reshaped = torch.empty_like(x_reshaped)
            softmax_kernel[(num_programs,)](
                x_ptr=x_reshaped,
                out_ptr=out_reshaped,
                n_elements=rows * cols,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            # Reshape back to original
            return out_reshaped.reshape(orig_shape)
        else:
            # Fallback to PyTorch for non-standard cases
            return x.softmax(dim=-1)

def replacement_func():
    return optimized_softmax