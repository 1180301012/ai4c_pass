import torch
import triton
import triton.language as tl

def pattern(x):
    t1 = torch.nn.functional.silu(x)
    t2 = torch.nn.functional.max_pool2d(t1, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return t2, t1

def replacement_args(x):
    return (x,)

@triton.jit
def silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """High-performance SILU kernel: output = x * sigmoid(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # SILU: x * sigmoid(x) = x / (1 + exp(-x))
    # Using stable sigmoid computation
    neg_x = -x
    # Clamp to avoid overflow
    neg_x = tl.where(neg_x > 88.7, 88.7, neg_x)
    exp_neg_x = tl.exp(neg_x)
    sigmoid_x = 1.0 / (1.0 + exp_neg_x)
    
    # SILU operation
    out = x * sigmoid_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def max_pool2d_kernel_5x5(
    input_ptr, output_ptr, 
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    """5x5 max pooling with padding 2, stride 1"""
    # Each program handles one output pixel
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)
    
    # Calculate input coordinates with padding
    h_in_center = h_out + 2  # padding 2
    w_in_center = w_out + 2  # padding 2
    
    # Initialize max value with -infinity
    max_val = -tl.float32('inf')
    
    # 5x5 window around center pixel
    for i in range(-2, 3):  # -2, -1, 0, 1, 2
        for j in range(-2, 3):  # -2, -1, 0, 1, 2
            h_in = h_in_center + i
            w_in = w_in_center + j
            
            # Check bounds
            if 0 <= h_in < H and 0 <= w_in < W:
                # Calculate global index
                offset = batch_idx * C * H * W + channel_idx * H * W + h_in * W + w_in
                val = tl.load(input_ptr + offset)
                if val > max_val:
                    max_val = val
    
    # Store output
    out_offset = batch_idx * C * H * W + channel_idx * H * W + h_out * W + w_out
    tl.store(output_ptr + out_offset, max_val)

@torch.fx.wrap
def optimized_silu_max_pool(x):
    """Optimized implementation of SILU followed by MaxPool2d"""
    device = x.device
    dtype = x.dtype
    shape = x.shape
    N, C, H, W = shape
    
    # Apply SILU
    x_contiguous = x.contiguous()
    silu_out = torch.empty_like(x_contiguous)
    
    # Launch SILU kernel
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    silu_kernel[(num_programs,)](
        x_ptr=x_contiguous,
        out_ptr=silu_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Apply MaxPool2d 5x5 with padding 2
    maxpool_out = torch.empty_like(silu_out)
    
    # Calculate grid dimensions for max pooling
    # Using smaller block sizes for better GPU occupancy
    BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W = 1, 1, 8, 8
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C  
    grid_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    max_pool2d_kernel_5x5[(grid_n, grid_c, grid_h, grid_w)](
        input_ptr=silu_out,
        output_ptr=maxpool_out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return maxpool_out, silu_out

def replacement_func():
    return optimized_silu_max_pool