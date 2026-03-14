import torch
import triton
import triton.language as tl

def pattern(in_7, tmp_9):
    # Residual connection addition
    tmp_10 = in_7 + tmp_9
    return tmp_10

def replacement_args(in_7, tmp_9):
    return (in_7, tmp_9)

@triton.jit
def residual_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Grid mapping: (N, C, H, W) -> (program_id_0, program_id_1, program_id_2, program_id_3)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Each thread handles a block of spatial positions
    n_start = pid_n * BLOCK_SIZE_HW
    c_start = pid_c * 1  # Process one channel at a time
    h_start = pid_h * BLOCK_SIZE_HW
    w_start = pid_w * BLOCK_SIZE_HW
    
    # Block boundaries
    n_end = min(n_start + BLOCK_SIZE_HW, N)
    c_end = min(c_start + 1, C)
    h_end = min(h_start + BLOCK_SIZE_HW, H)
    w_end = min(w_start + BLOCK_SIZE_HW, W)
    
    # Process the block
    for nh in range(n_start, n_end):
        for c in range(c_start, c_end):
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    # Load both input values
                    idx = nh * C * H * W + c * H * W + h * W + w
                    x_val = tl.load(x_ptr + idx)
                    y_val = tl.load(y_ptr + idx)
                    
                    # Perform addition
                    result = x_val + y_val
                    
                    # Store result
                    tl.store(output_ptr + idx, result)

@torch.fx.wrap
def optimized_residual_add(in_7, tmp_9):
    if in_7.shape != tmp_9.shape:
        raise ValueError(f"Shape mismatch: {in_7.shape} vs {tmp_9.shape}")
    
    N, C, H, W = in_7.shape
    output = torch.empty_like(in_7)
    
    # Use 8x8 block size for spatial dimensions
    BLOCK_SIZE_HW = 8
    
    # Calculate grid size
    grid_n = (N + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    grid_c = (C + 1 - 1) // 1  # One channel per program
    grid_h = (H + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    grid_w = (W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    residual_add_kernel[(grid_n, grid_c, grid_h, grid_w)](
        in_7,
        tmp_9,
        output,
        N, C, H, W,
        BLOCK_SIZE_HW
    )
    
    return output

def replacement_func():
    return optimized_residual_add