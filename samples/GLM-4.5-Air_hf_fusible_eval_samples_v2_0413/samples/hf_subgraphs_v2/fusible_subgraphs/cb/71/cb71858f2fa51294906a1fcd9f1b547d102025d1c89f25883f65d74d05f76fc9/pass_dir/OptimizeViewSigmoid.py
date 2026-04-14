import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Efficient sigmoid kernel using Triton"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Using exp approximation for better performance
    x_neg = -x
    exp_x = tl.exp(tl.where(x_neg > 0, x_neg, 0))  # exp(-x) for x > 0
    sigmoid = 1.0 / (1.0 + exp_x)
    
    # Store result
    tl.store(out_ptr + offsets, sigmoid, mask=mask)

@torch.fx.wrap
def efficient_sigmoid_with_reshape(x, target_shape):
    """Efficient sigmoid with reshape operation in one kernel"""
    # Create output tensor with target shape
    out = torch.empty(target_shape, dtype=x.dtype, device=x.device)
    
    n_elements = x.numel()
    
    # Use efficient block size for sigmoid computation
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(conv_output):
    """Match: view -> sigmoid pattern"""
    tmp_3 = conv_output.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    return tmp_4

def replacement_args(conv_output):
    """Extract arguments for replacement"""
    return (conv_output,)

def replacement_func():
    """Return the optimized function that combines reshape and sigmoid"""
    def optimized_func(conv_output):
        target_shape = (1, 2, 8, 8)
        return efficient_sigmoid_with_reshape(conv_output, target_shape)
    
    return optimized_func