import torch
import triton
import triton.language as tl

def pattern(in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    return tmp_3

def replacement_args(in_1, in_2):
    return (in_1, in_2)



@triton.jit
def broadcast_mul_sigmoid_kernel(
    in_1_ptr, 
    in_2_ptr,
    out_ptr,
    n_elements,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor data (in_1)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized channel calculation - compute channel directly from offset
    idx = offsets
    channel = (idx // (height * width)) % channels
    
    # Load sigmoid coefficients directly
    sigmoid_coef = tl.load(in_2_ptr + channel, mask=channel < channels, other=0.0)
    
    # Apply sigmoid and multiply
    sigmoid_val = tl.sigmoid(sigmoid_coef)
    out = in_1 * sigmoid_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def broadcast_mul_sigmoid_kernel_wrapper(in_1, in_2):
    batch_size, channels, height, width = in_1.shape
    N = in_1.numel()
    
    # Optimize block size based on tensor size
    if width > 16:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 512
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_1)
    
    # Launch kernel with tensor shapes as arguments
    broadcast_mul_sigmoid_kernel[(num_programs,)](
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        n_elements=N,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return broadcast_mul_sigmoid_kernel_wrapper