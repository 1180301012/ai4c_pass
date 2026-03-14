import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Pattern to match: ReLU followed by Sigmoid
    This matches the exact computation in model.py
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1

def replacement_args(in_0):
    """Extract input tensor for the replacement function"""
    return (in_0,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_relu_sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes sigmoid(relu(x)) in a single pass
    """
    # Calculate the starting position for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    x_relu = tl.maximum(x, 0.0)
    
    # Apply Sigmoid: 1 / (1 + exp(-x))
    x_sigmoid = tl.sigmoid(x_relu)
    
    # Store output
    tl.store(output_ptr + offsets, x_sigmoid, mask=mask)

@torch.fx.wrap
def fused_relu_sigmoid(input_tensor):
    """
    Wrapper function that launches the fused ReLU+Sigmoid kernel
    """
    # Get total number of elements
    n_elements = input_tensor.numel()
    
    # Allocate output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate grid size
    def grid(meta):
        return ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    # Launch kernel
    fused_relu_sigmoid_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
    )
    
    return output

def replacement_func():
    """Return the replacement function (not a call to it)"""
    return fused_relu_sigmoid