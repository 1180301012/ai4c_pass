import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_expand_kernel(
    sigmoid_input_ptr,    # in_2: [1, 1, 2048]
    expand_output_ptr,    # expanded: [1, 2048, H, W]
    n_elements,           # total elements in output
    H: tl.constexpr,      # spatial height
    W: tl.constexpr,      # spatial width
    C: tl.constexpr,      # channels (2048)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert linear offset to 4D coordinates
    offset = offsets
    h = offset % H
    offset = offset // H
    w = offset % W  
    offset = offset // W
    c = offset % C
    
    # Load sigmoid weight for this channel
    sigmoid_weight = tl.load(sigmoid_input_ptr + c, mask=c < C, other=0.0)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-sigmoid_weight))
    
    # Store expanded sigmoid value for all spatial positions of this channel
    tl.store(expand_output_ptr + offsets, sigmoid_val, mask=mask)

@torch.fx.wrap
def optimized_sigmoid_expand(sigmoid_input, target_tensor):
    """
    Optimized sigmoid + view + expand operation
    Input: [1, 1, 2048] -> Output: same shape as target_tensor
    """
    target_shape = target_tensor.shape
    
    # Extract shape dimensions - target_shape should be [1, 2048, H, W]
    batch, channels, height, width = target_shape[0], target_shape[1], target_shape[2], target_shape[3]
    
    # Output has same shape as target
    out = torch.empty_like(target_tensor)
    
    n_elements = batch * channels * height * width
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    sigmoid_expand_kernel[(num_programs,)](
        sigmoid_input_ptr=sigmoid_input,
        expand_output_ptr=out,
        n_elements=n_elements,
        H=height,
        W=width,
        C=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(sigmoid_input, target_tensor):
    """Match: sigmoid -> view -> expand sequence"""
    sigmoid_result = sigmoid_input.sigmoid()
    viewed_result = sigmoid_result.view(1, -1, 1, 1)
    expanded_result = viewed_result.expand_as(target_tensor)
    return expanded_result

def replacement_args(sigmoid_input, target_tensor):
    return (sigmoid_input, target_tensor)

def replacement_func():
    return optimized_sigmoid_expand