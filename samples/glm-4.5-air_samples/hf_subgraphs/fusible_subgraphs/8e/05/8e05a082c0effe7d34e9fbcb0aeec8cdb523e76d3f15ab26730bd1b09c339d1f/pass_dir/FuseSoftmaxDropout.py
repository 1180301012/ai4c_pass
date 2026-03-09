import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_12 = torch.nn.functional.softmax(x, dim=-1)
    tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False)
    return tmp_13

def replacement_args(x):
    return (x,)

@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute softmax using Triton's built-in operations
    # For numerical stability, we can use the built-in softmax
    out = tl.softmax(x, dim=-1)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap  
def optimized_softmax(x):
    """
    Optimized softmax that implicitly handles dropout with p=0.0
    (dropout with p=0.0 is just identity, so we can skip it entirely)
    """
    if x.dim() == 4:  # Common case for attention: (batch*heads, channels, height, width)
        # Reshape to 2D for softmax computation, then reshape back
        original_shape = x.shape
        batch_heads = original_shape[0]
        channels = original_shape[1]
        spatial_elements = original_shape[2] * original_shape[3]
        
        # Reshape to (batch_heads * channels, spatial_elements) for softmax on last dim
        x_reshaped = x.reshape(batch_heads * channels, spatial_elements)
        
        N = x_reshaped.numel()
        BLOCK_SIZE = 1024
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out = torch.empty_like(x_reshaped)
        
        softmax_kernel[(num_programs,)](
            x_ptr=x_reshaped,
            out_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Reshape back to original dimensions
        return out.reshape(original_shape)
    else:
        # Fallback for other dimensions
        N = x.numel()
        BLOCK_SIZE = 1024
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out = torch.empty_like(x)
        
        softmax_kernel[(num_programs,)](
            x_ptr=x,
            out_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out

def replacement_func():
    return optimized_softmax