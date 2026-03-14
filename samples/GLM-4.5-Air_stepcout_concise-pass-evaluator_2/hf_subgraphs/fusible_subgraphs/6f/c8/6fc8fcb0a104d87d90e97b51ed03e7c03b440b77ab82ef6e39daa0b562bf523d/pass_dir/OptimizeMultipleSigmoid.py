import torch
import triton
import triton.language as tl

def sigmoid_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Sigmoid kernel for computing sigmoid on GPU"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid using stable formula
    # sigmoid(x) = 1 / (1 + exp(-x))
    x_neg = -x
    exp_x = tl.exp(x_neg)
    sigmoid = 1.0 / (1.0 + exp_x)
    
    # Store result
    tl.store(output_ptr + offsets, sigmoid, mask=mask)

@torch.fx.wrap
def triton_sigmoid(x):
    """Triton-accelerated sigmoid function"""
    n_elements = x.numel()
    if n_elements == 0:
        return torch.empty_like(x)
    
    # Use optimal block size for GPU
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def pattern(in_tensor):
    """Pattern: Compute sigmoid on the input tensor"""
    return torch.nn.functional.sigmoid(in_tensor)

def replacement_args(in_tensor):
    """Extract the input tensor for sigmoid computation"""
    return (in_tensor,)

def replacement_func():
    """Return the optimized sigmoid function"""
    return triton_sigmoid