import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Matches the interpolation operation with size (40, 40) and mode 'nearest'"""
    return torch.nn.functional.interpolate(input_tensor, size=(40, 40), mode='nearest')

def replacement_args(input_tensor):
    """Extracts the input tensor for the replacement"""
    return (input_tensor,)

@triton.jit
def upsample_nearest2_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program id
    pid = tl.program_id(0)
    # Start at [pid * BLOCK_SIZE, pid * BLOCK_SIZE + BLOCK_SIZE) for rows
    start_row = pid * BLOCK_SIZE
    # Loop through the rows in this block
    for r in range(start_row, H_out):
        for c in range(0, W_out):
            # Map to input coordinates (nearest neighbor)
            input_r = r // 2
            input_c = c // 2
            # Load input value (fixed input layout)
            input_val = tl.load(input_ptr + (input_r * W_in + input_c), mask=tl.arange(0, 1) < 1)
            # Store into output
            tl.store(output_ptr + (r * W_out + c), input_val, mask=tl.arange(0, 1) < 1)

@torch.fx.wrap
def upsample_nearest2(input_tensor):
    B, C, H_in, W_in = input_tensor.shape
    H_out = 40
    W_out = 40
    output = torch.empty_like(input_tensor)
    # Launch kernel with 1D grid
    upsample_nearest2_kernel[(H_out + BLOCK_SIZE - 1) // BLOCK_SIZE](input_ptr=input_tensor,
        output_ptr=output,
        B=B,
        C=C,
        H_in=H_in,
        W_in=W_in,
        H_out=H_out,
        W_out=W_out,
        BLOCK_SIZE=128,
    )
    return output

def replacement_func():
    return upsample_nearest2