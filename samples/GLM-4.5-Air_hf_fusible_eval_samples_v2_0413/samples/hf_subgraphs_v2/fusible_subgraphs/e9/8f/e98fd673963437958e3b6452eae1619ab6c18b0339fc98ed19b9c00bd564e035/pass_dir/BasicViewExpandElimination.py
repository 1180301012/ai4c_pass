import torch
import triton
import triton.language as tl

@triton.jit
def compute_broadcast_multiply_kernel(
    sigmoid_flat_ptr,  # Shape [2048] - flattened sigmoid values
    in_1_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * channels * height * width
    
    # Load in_1 value
    in_1_val = tl.load(in_1_ptr + offsets, mask=mask)
    
    # Calculate coordinates: [b, c, h, w]
    linear_idx = offsets
    c = (linear_idx // (height * width)) % channels  # Channel index
    h = (linear_idx // width) % height              # Row index
    w = linear_idx % width                          # Column index
    
    # Load sigmoid value for this channel
    sigmoid_val = tl.load(sigmoid_flat_ptr + c, mask=c < channels)
    
    # Element-wise multiplication
    result = in_1_val * sigmoid_val
    
    # Store result  
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def compute_broadcast_multiply(sigmoid_input, in_1):
    """Compute broadcast multiplication using Triton kernel"""
    # Create output tensor using allowed operation
    out = torch.empty_like(in_1)
    
    # Extract tensor dimensions - in_1 has shape [batch, channels, height, width]
    batch_size = in_1.shape[0]
    channels = in_1.shape[1] 
    height = in_1.shape[2]
    width = in_1.shape[3]
    
    # Launch Triton kernel - sigmoid_input has shape [1, 1, 2048]
    n_elements = in_1.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    compute_broadcast_multiply_kernel[(num_programs,)](
        sigmoid_flat_ptr=sigmoid_input,
        in_1_ptr=in_1,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_2, in_1):
    # Match: in_2.sigmoid() -> view(1,-1,1,1) -> expand_as(in_1) -> multiply with in_1
    
    sigmoid_val = in_2.sigmoid()
    view_op = sigmoid_val.view(1, -1, 1, 1)
    expand_op = view_op.expand_as(in_1)
    result = in_1 * expand_op
    
    return result

def replacement_args(in_2, in_1):
    return (in_2, in_1)

def replacement_func():
    return compute_broadcast_multiply