import torch
import triton
import triton.language as tl

# Pattern matching function - matches ReLU followed by dropout2d with training=False
# dropout2d with training=False and inplace=False just returns the input unchanged
def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return tmp_1, tmp_0

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def relu_dropout2d_fused_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU inplace semantics: max(0, x)
    relu_out = tl.where(x > 0, x, 0.0)
    
    # Store to output
    tl.store(out_ptr + offsets, relu_out, mask=mask)

@torch.fx.wrap
def relu_dropout2d_fused(in_0):
    # Get total number of elements
    n_elements = in_0.numel()
    
    # Choose block size based on tensor size for optimal occupancy
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor (same shape as input)
    out = torch.empty_like(in_0)
    
    # Launch kernel
    relu_dropout2d_fused_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return (dropout_out, relu_out) - both are the same tensor in this case
    # Since dropout2d with training=False returns input unchanged
    return out, out

def replacement_func():
    return relu_dropout2d_fused