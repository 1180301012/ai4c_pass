import torch
import triton
import triton.language as tl

# Pattern matching function for the addition + dropout2d stream
def pattern(in_3, in_4):
    # Match the addition followed by dropout2d exactly as in the model
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4

# Argument extraction function
def replacement_args(in_3, in_4):
    return (in_3, in_4)

# Triton kernel for fused add + dropout2d (evaluation mode)
@triton.jit
def add_dropout2d_kernel(
    x_ptr,          # first input tensor [N, C, H, W]
    y_ptr,          # second input tensor [N, C, H, W] 
    out_ptr,        # output tensor [N, C, H, W]
    n_elements: tl.constexpr,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition and apply dropout (in evaluation mode, dropout is identity)
    result = x + y
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

# Fused addition + dropout2d wrapper (evaluation mode)
@torch.fx.wrap
def fused_add_dropout2d(x, y):
    # Get tensor shape info
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    
    n_batch, n_channels, height, width = x.shape
    n_elements = n_batch * n_channels * height * width
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Set block size and launch grid
    BLOCK_SIZE = 1024  # Optimal block size for good occupancy
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # In evaluation mode, dropout2d is just identity operation
    # So we just do fused addition
    add_dropout2d_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return fused_add_dropout2d