import torch
import triton
import triton.language as tl


@triton.jit
def relu_view_unsqueeze_kernel_dim1(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs ReLU activation on input.
    The view and unsqueeze are handled via tensor operations in Python.
    """
    # Each program processes a block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Load input - apply relu (max(0, x))
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    relu_x = tl.where(x > 0, x, 0.0)
    
    # Store relu output
    tl.store(output_ptr + offsets, relu_x, mask=mask)


@torch.fx.wrap
def relu_view_unsqueeze_dim1_wrapper(x):
    """
    Wrapper function for fused relu + view + unsqueeze operation.
    Uses Triton kernel for relu, then PyTorch for view/unsqueeze.
    """
    # Get total number of elements
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output with same shape as input
    output = torch.empty_like(x)
    
    # Run the Triton kernel for relu
    relu_view_unsqueeze_kernel_dim1[(num_programs,)](
        x,
        output,
        n_elements,
        BLOCK_SIZE,
    )
    
    # Now apply view and unsqueeze using PyTorch
    # Original input: [1, 512, 64, 64]
    # After view: [1, 512, 4096]
    # After unsqueeze: [1, 1, 512, 4096]
    dim0 = x.shape[0]
    tmp = output.view(dim0, 512, 4096)
    tmp = tmp.unsqueeze(1)
    
    # Return both the reshaped tensor and the relu output
    return tmp, output


def pattern(in_0):
    """
    Match the pattern: relu -> view(1,512,4096) -> unsqueeze
    Returns both the unsqueezed tensor and the relu output.
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.view(1, 512, 4096)
    tmp_2 = tmp_1.unsqueeze(1)
    return (tmp_2, tmp_0)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return relu_view_unsqueeze_dim1_wrapper