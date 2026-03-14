import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return (tmp_1, tmp_0)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_silu_maxpool_kernel(
    input_ptr,
    silu_out_ptr,
    maxpool_out_ptr,
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
    w_out = pid_w * BLOCK_SIZE_H
    
    # Initialize for SILU computation
    element_offset = (n_offset * C + c_offset) * H * W + h_out * W + w_out
    if element_offset < N * C * H * W:
        x = tl.load(input_ptr + element_offset)
        # SILU: x * sigmoid(x) = x / (1 + exp(-x))
        silu_val = x / (1.0 + tl.exp(-x))
        tl.store(silu_out_ptr + element_offset, silu_val)
    
    # Max Pool computation
    if n_offset < N and c_offset < C and h_out < H and w_out < W:
        # Calculate input coordinates with padding
        h_in = h_out * SH + PH
        w_in = w_out * SW + PW
        
        max_val = -float('inf')
        
        # Loop over kernel dimensions
        for kh in range(KH):
            for kw in range(KW):
                # Calculate actual input position
                h_curr = h_in + kh
                w_curr = w_in + kw
                
                # Check bounds
                if 0 <= h_curr < H and 0 <= w_curr < W:
                    offset = (n_offset * C + c_offset) * H * W + h_curr * W + w_curr
                    val = tl.load(input_ptr + offset)
                    if val > max_val:
                        max_val = val
        
        # Calculate max pool output offset
        maxpool_offset = (n_offset * C + c_offset) * H * W + h_out * W + w_out
        tl.store(maxpool_out_ptr + maxpool_offset, max_val)

@torch.fx.wrap
def fused_silu_maxpool2d(x):
    if x.numel() == 0:
        silu_out = torch.empty_like(x)
        maxpool_out = torch.empty_like(x)
        return maxpool_out, silu_out
    
    N, C, H, W = x.shape
    
    # Choose block sizes optimized for our input patterns
    BLOCK_SIZE_N = 1      # Process one batch at a time
    BLOCK_SIZE_C = 64     # Process chunks of channels
    BLOCK_SIZE_H = 8      # Process spatial blocks
    
    # Calculate grid dimensions
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (W + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    
    # Create output tensors
    silu_out = torch.empty_like(x)
    maxpool_out = torch.empty_like(x)
    
    # Launch the fused kernel
    fused_silu_maxpool_kernel[(grid_n, grid_c, grid_h, grid_w)](
        input_ptr=x,
        silu_out_ptr=silu_out,
        maxpool_out_ptr=maxpool_out,
        N=N, C=C, H=H, W=W,
        KH=5, KW=5,  # kernel size 5x5
        SH=1, SW=1,  # stride 1
        PH=2, PW=2,  # padding 2
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    return maxpool_out, silu_out

def replacement_func():
    return fused_silu_maxpool2d