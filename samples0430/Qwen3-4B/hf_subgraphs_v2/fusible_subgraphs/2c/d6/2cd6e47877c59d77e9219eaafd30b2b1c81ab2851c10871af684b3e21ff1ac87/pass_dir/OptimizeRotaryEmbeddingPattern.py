import torch
import triton
import triton.language as tl

def pattern(in_3, in_1, in_5):
    # Split the tensor
    odd = in_3[..., 1::2]
    even = in_3[..., ::2]
    
    # Create negative odd part
    odd_neg = -odd
    
    # Stack the two parts along the last dimension
    stacked = torch.stack([odd_neg, even], dim=-1)
    
    # Reshape to (1, N, H, W)
    reshaped = stacked.reshape(1, *stacked.shape[-3:])
    
    # Multiply by in_5
    mult = reshaped * in_5
    
    # Add the scaled in_3 (from in_3 * in_1)
    return (in_3 * in_1) + mult
def replacement_args(in_3, in_1, in_5):
    return (in_3, in_1, in_5)

@triton.jit
def optimized_rotary_kernel(
    in_3_ptr,
    in_1_ptr,
    in_5_ptr,
    out_ptr,
    N: tl.int32,
    H: tl.int32,
    W: tl.int32,
    BLOCK_SIZE: tl.constexpr = 128
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE, 1)
    mask = offsets < (N * H * W)
    
    # Load tensors
    in_3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_5 = tl.load(in_5_ptr + offsets, mask=mask, other=0.0)
    
    # Generate odd and even parts
    indices = offsets % 2
    odd = tl.where(indices == 1, in_3, 0.0)
    even = tl.where(indices == 0, in_3, 0.0)
    
    # Create negative for odd
    odd_neg = -odd
    
    # Stack parts
    stack = tl.stack([odd_neg, even], dim=-1)
    
    # Reshape
    reshaped = stack.reshape(1, N, H, W)  # Just for demonstration
    
    # Transform
    mult = reshaped * in_5
    
    # Compute and store result
    out = (in_3 * in_1) + mult
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_rotary_function(in_3, in_1, in_5):
    N = in_3.shape[1]
    H = in_3.shape[2]
    W = in_3.shape[3]
    num_elements = N * H * W
    
    out = torch.empty_like(in_3)
    
    # Calculate the number of blocks
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    optimized_rotary_kernel[(num_blocks,)](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_5_ptr=in_5,
        out_ptr=out,
        N=N,
        H=H,
        W=W,
        BLOCK_SIZE=128
    )
    
    return out
def replacement_func():
    return optimized_rotary_function