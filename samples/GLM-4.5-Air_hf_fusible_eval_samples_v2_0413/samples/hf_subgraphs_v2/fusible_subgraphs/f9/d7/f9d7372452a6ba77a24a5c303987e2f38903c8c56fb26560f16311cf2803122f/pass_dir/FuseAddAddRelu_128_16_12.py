import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_add_relu_kernel(
    x_ptr,
    y_ptr, 
    z_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations: x + y + z then ReLU
    result = x + y + z
    result = tl.maximum(result, 0.0)  # ReLU operation
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_add_add_relu(x, y, z):
    """Fused addition of three tensors followed by ReLU activation"""
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same properties as inputs
    out = torch.empty_like(x)
    
    # Launch the fused kernel
    fused_add_add_relu_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_3, in_0, in_2):
    """Match the exact dataflow from model: in_3 += in_0; in_4 = in_3; in_4 += in_2; tmp_0 = in_4; tmp_2 = relu(tmp_0, inplace=True)"""
    # Create copies to simulate in-place operations
    in_3_new = in_3 + in_0  # simulates in_3 += in_0
    in_4 = in_3_new        # simulates in_4 = in_3
    in_4_new = in_4 + in_2  # simulates in_4 += in_2
    tmp_0 = in_4_new
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)  # exact match with inplace=True
    
    return tmp_2





def replacement_args(a, b, c):
    """Extract the three input tensors for the fused operation"""
    return (a, b, c)

def replacement_func():
    """Return the fused kernel function"""
    return fused_add_add_relu