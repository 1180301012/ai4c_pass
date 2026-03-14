import torch
import triton
import triton.language as tl

# Pattern matching function - matches ReLU followed by Sigmoid
def pattern(x):
    """
    Matches ReLU(x, inplace=True) followed by Sigmoid(relu_result)
    This matches the computation pattern in both target graphs.
    """
    relu_out = torch.nn.functional.relu(x, inplace=True)
    sigmoid_out = torch.sigmoid(relu_out)
    return sigmoid_out

# Argument extraction function
def replacement_args(x):
    """Extract the input tensor needed for the replacement"""
    return (x,)

# Triton kernel implementation
@triton.jit
def fused_relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Sigmoid kernel using Triton"""
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: ReLU then Sigmoid
    # ReLU: max(0, x)
    relu_out = tl.maximum(x, 0.0)
    # Sigmoid: 1 / (1 + exp(-relu_out))
    sigmoid_out = 1.0 / (1.0 + tl.exp(-relu_out))
    
    # Store result
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)

# Kernel wrapper function
@torch.fx.wrap
def fused_relu_sigmoid_wrapper(x):
    """Launch the fused ReLU+Sigmoid kernel on GPU"""
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Create output tensor with same shape and dtype as input
    out = torch.empty_like(x)
    
    # Launch kernel with autotune for optimal block size
    fused_relu_sigmoid_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024  # Will be autotuned by framework
    )
    
    return out

# Replacement function (returns the wrapper function)
def replacement_func():
    """Return the fused kernel wrapper function"""
    return fused_relu_sigmoid_wrapper