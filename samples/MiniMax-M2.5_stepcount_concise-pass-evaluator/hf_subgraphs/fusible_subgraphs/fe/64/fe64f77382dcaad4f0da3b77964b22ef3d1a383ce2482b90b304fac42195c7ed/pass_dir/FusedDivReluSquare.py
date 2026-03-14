import torch
import triton
import triton.language as tl

# The pattern to match: division -> relu -> square
# Note: The constant 11.313708498984761 is sqrt(128) which is used for normalization
DIV_CONSTANT = 11.313708498984761

def pattern(in_0):
    """
    Match the computation pattern: (in_0 / const) -> relu -> square
    This pattern appears in RTMPose for attention score normalization
    """
    tmp_0 = in_0 / DIV_CONSTANT
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = torch.square(tmp_1)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

# Autotune configuration for optimal block sizes
@triton.autotune(
    configs=[
        # Different block sizes for various tensor sizes
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: (x / const) -> relu -> square
    
    This fuses three operations into one kernel:
    1. Division by constant
    2. ReLU activation (max(0, x))
    3. Square (x^2)
    
    Combined: square(relu(x / const))
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Operation 1: Division by constant
    x = x / DIV_CONSTANT
    
    # Operation 2: ReLU (max with 0)
    x = tl.maximum(x, 0.0)
    
    # Operation 3: Square
    x = x * x
    
    # Store result
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_div_relu_square_kernel_wrapper(in_0):
    """
    Wrapper function that launches the fused Triton kernel.
    Handles 3D tensors by flattening and processing all elements.
    """
    # Flatten the tensor to 1D for processing
    original_shape = in_0.shape
    x_flat = in_0.flatten()
    n_elements = x_flat.numel()
    
    # Determine block size based on total elements
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor with same shape and device
    out_flat = torch.empty_like(x_flat)
    
    # Launch kernel
    fused_div_relu_square_kernel[(num_programs,)](
        x_ptr=x_flat,
        out_ptr=out_flat,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to original shape
    return out_flat.view(original_shape)

def replacement_func():
    return fused_div_relu_square_kernel_wrapper