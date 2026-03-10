import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Transpose the last two dimensions
    return input_tensor.transpose(-2, -1)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,      # 1 (batch)
    C: tl.constexpr,      # 16 (channels)
    H: tl.constexpr,      # 196 (height)
    W: tl.constexpr,      # 48 (width)
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # Get program IDs
    n = tl.program_id(0)  # batch (1)
    c = tl.program_id(1)  # channel (16)
    h = tl.program_id(2)  # height (196)
    w = tl.program_id(3)  # width (48)
    
    # Calculate input and output offsets
    input_offset = n * (C * H * W) + c * (H * W) + h * W + w
    output_offset = n * (C * W * H) + c * (W * H) + w * H + h
    
    # Bounds checking with nested ifs
    if n >= N:
        return
    if c >= C:
        return
    if h >= H:
        return
    if w >= W:
        return
    
    # Load from input and store to output (swapped indices)
    value = tl.load(input_ptr + input_offset)
    tl.store(output_ptr + output_offset, value)

@torch.fx.wrap
def optimized_transpose(input_tensor):
    # Get tensor shape info
    N = input_tensor.shape[0]  # 1
    C = input_tensor.shape[1]  # 16
    H = input_tensor.shape[2]  # 196
    W = input_tensor.shape[3]  # 48
    
    # Create output tensor with swapped dimensions
    output = torch.empty([N, C, W, H], dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid dimensions
    grid = (
        (N + 63) // 64,      # batch
        (C + 63) // 64,      # channels  
        (H + 63) // 64,      # height
        (W + 63) // 64       # width
    )
    
    transpose_kernel[grid](
        input_tensor, output,
        N, C, H, W,
        64, 64
    )
    
    return output

def replacement_func():
    return optimized_transpose