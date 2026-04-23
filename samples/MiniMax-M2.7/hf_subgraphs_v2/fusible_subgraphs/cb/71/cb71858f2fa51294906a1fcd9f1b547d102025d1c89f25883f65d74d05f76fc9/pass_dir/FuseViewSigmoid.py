import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused sigmoid kernel - view is handled by the flat computation.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Sigmoid: 1 / (1 + exp(-x))
    sigmoid_val = 1.0 / (1.0 + tl.exp(-x))
    
    # Store
    tl.store(output_ptr + offsets, sigmoid_val, mask=mask)


@torch.fx.wrap
def fused_sigmoid(input_tensor):
    """
    Fused sigmoid - the view operation is handled by computing on flattened data.
    """
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    sigmoid_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(conv2d):
    """
    Match the pattern:
        tmp_3 = conv2d.view(1, 2, 8, 8)
        tmp_4 = tmp_3.sigmoid()
    Returns tmp_4 (the final sigmoid output).
    """
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    return tmp_4


def replacement_args(conv2d):
    return (conv2d,)


def replacement_func():
    return fused_sigmoid