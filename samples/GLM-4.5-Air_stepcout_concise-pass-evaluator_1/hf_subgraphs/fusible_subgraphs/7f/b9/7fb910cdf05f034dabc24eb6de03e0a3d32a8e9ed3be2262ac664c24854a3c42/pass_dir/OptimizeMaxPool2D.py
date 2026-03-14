import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return tmp_1

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def maxpool2d_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    KH, KW,  # kernel height and width
    SH, SW,  # stride height and width
    PH, PW,  # padding height and width
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Calculate program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate batch and channel offsets
    n_offset = pid_n * BLOCK_SIZE_N
    c_offset = pid_c * BLOCK_SIZE_C
    
    # Calculate output coordinates
    h_out = pid_h * BLOCK_SIZE_H
    w_out = pid_w * BLOCK_SIZE_H  # assuming square blocks
    
    # Calculate input coordinates with padding
    h_in = h_out * SH + PH
    w_in = w_out * SW + PW
    
    # Initialize max value
    max_val = -float('inf')
    
    # Loop over kernel dimensions
    for kh in range(KH):
        for kw in range(KW):
            # Calculate actual input position
            h_curr = h_in + kh
            w_curr = w_in + kw
            
            # Check bounds
            if 0 <= h_curr < H and 0 <= w_curr < W and n_offset < N and c_offset < C:
                # Calculate memory offset
                offset = (n_offset * C + c_offset) * H * W + h_curr * W + w_curr
                val = tl.load(input_ptr + offset)
                if val > max_val:
                    max_val = val
    
    if n_offset < N and c_offset < C and h_out < H and w_out < W:
        # Calculate output offset
        out_offset = (n_offset * C + c_offset) * H * W + h_out * W + w_out
        tl.store(output_ptr + out_offset, max_val)

@torch.fx.wrap
def triton_maxpool2d(x, kernel_size=5, stride=1, padding=2):
    # For this specific pattern: kernel_size=5, stride=1, padding=2
    # This preserves spatial dimensions since (5-2*2)/1 + 1 = 1
    
    if x.numel() == 0:
        return torch.empty_like(x)
    
    N, C, H, W = x.shape
    
    # Choose block sizes that work well for our input dimensions
    BLOCK_SIZE_N = 1  # Process one batch at a time
    BLOCK_SIZE_C = 64  # Process multiple channels together
    BLOCK_SIZE_H = 8   # Process spatial blocks
    
    # Calculate grid dimensions
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    
    output = torch.empty_like(x)
    
    maxpool2d_kernel[(grid_n, grid_c, grid_h, grid_w)](
        input_ptr=x,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        KH=kernel_size, KW=kernel_size,
        SH=stride, SW=stride,
        PH=padding, PW=padding,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return output

def replacement_func():
    return lambda x: triton_maxpool2d(x, kernel_size=5, stride=1, padding=2)