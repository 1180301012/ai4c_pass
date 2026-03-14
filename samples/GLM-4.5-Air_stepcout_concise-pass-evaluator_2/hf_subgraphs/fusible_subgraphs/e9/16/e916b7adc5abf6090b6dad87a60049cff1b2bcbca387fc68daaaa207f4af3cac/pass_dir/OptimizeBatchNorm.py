import torch
import triton
import triton.language as tl

def pattern(tmp_10, in_3, in_4, in_6, in_5):
    # Batch norm operation
    tmp_11 = torch.nn.functional.batch_norm(tmp_10, in_3, in_4, in_6, in_5, False, 0.1, 1e-05)
    return tmp_11

def replacement_args(tmp_10, in_3, in_4, in_6, in_5):
    return (tmp_10, in_3, in_4, in_6, in_5)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N, C, H, W,
    momentum: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Grid mapping: (N, C) -> (program_id_0, program_id_1)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Load batch norm parameters
    running_mean = tl.load(running_mean_ptr + pid_c, mask=pid_c < 64)
    running_var = tl.load(running_var_ptr + pid_c, mask=pid_c < 64)
    weight = tl.load(weight_ptr + pid_c, mask=pid_c < 64)
    bias = tl.load(bias_ptr + pid_c, mask=pid_c < 64)
    
    # Calculate mean and variance for batch normalization
    # In training mode, we use running statistics (as indicated by training=False)
    # But the kernel should compute the normalized output
    
    # Each thread handles a block of spatial positions
    n_start = pid_n * BLOCK_SIZE_N
    c_start = pid_c * BLOCK_SIZE_C
    
    n_end = min(n_start + BLOCK_SIZE_N, N)
    c_end = min(c_start + BLOCK_SIZE_C, C)
    
    # Precompute normalization constants
    std = tl.sqrt(running_var + eps)
    inv_std = 1.0 / std
    
    # Process the block
    for nh in range(n_start, n_end):
        for c_local in range(c_start, c_end):
            # Process spatial block
            for h_tile in range(0, H, BLOCK_SIZE_HW):
                for w_tile in range(0, W, BLOCK_SIZE_HW):
                    h_end = min(h_tile + BLOCK_SIZE_HW, H)
                    w_end = min(w_tile + BLOCK_SIZE_HW, W)
                    
                    for h in range(h_tile, h_end):
                        for w in range(w_tile, w_end):
                            # Load input value
                            input_idx = nh * C * H * W + c_local * H * W + h * W + w
                            x = tl.load(input_ptr + input_idx)
                            
                            # Batch normalization formula: (x - mean) * (weight / std) + bias
                            norm_x = (x - running_mean) * inv_std
                            output_val = norm_x * weight + bias
                            
                            # Store result
                            output_idx = nh * C * H * W + c_local * H * W + h * W + w
                            tl.store(output_ptr + output_idx, output_val)

@torch.fx.wrap
def optimized_batch_norm(tmp_10, in_3, in_4, in_6, in_5):
    input_shape = tmp_10.shape
    N, C, H, W = input_shape
    
    # Output has same shape as input
    out_shape = input_shape
    output = torch.empty(out_shape, dtype=torch.float32, device=tmp_10.device)
    
    # Use small block sizes for better parallelism
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 64  # Process all channels at once
    BLOCK_SIZE_HW = 8 * 8  # 8x8 spatial tiling
    
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    batch_norm_kernel[(grid_n, grid_c)](
        tmp_10,
        in_3,
        in_4,
        in_6,
        in_5,
        output,
        N, C, H, W,
        0.1,  # momentum (not used in inference)
        1e-05,  # epsilon
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_HW
    )
    
    return output

def replacement_func():
    return optimized_batch_norm