import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern function - match just flatten operation on a tensor.
    """
    return in_0.flatten(1, -1)


def replacement_args(in_0):
    return (in_0,)


# Triton kernel for flatten operation
@triton.jit
def flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate which elements this program block handles
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load from input tensor
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Store to output tensor
    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def flatten_kernel_wrapper(in_0):
    """
    Wrapper function for flatten kernel.
    """
    input_shape = in_0.shape
    n_elements = in_0.numel()
    
    # Calculate output shape: flatten from dimension 1 to end
    # Input: [B, C, H, W] -> Output: [B, C*H*W]
    batch_size = input_shape[0]
    output_features = n_elements // batch_size
    
    # Create output tensor
    out = torch.empty((batch_size, output_features), dtype=in_0.dtype, device=in_0.device)
    
    # Configure kernel launch
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    flatten_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return flatten_kernel_wrapper