import torch
import triton
import triton.language as tl

def pattern(x):
    return torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)

@triton.jit
def avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W, 
    pool_size: tl.constexpr, stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # channel dimension
    
    # Start position for this batch and channel
    start_m = pid_m * BLOCK_SIZE_M
    end_m = min(start_m + BLOCK_SIZE_M, N)
    start_n = pid_n * BLOCK_SIZE_N
    end_n = min(start_n + BLOCK_SIZE_N, C)
    
    # Input size after pooling
    H_out = H // stride
    W_out = W // stride
    
    for m in range(start_m, end_m):
        for n in range(start_n, end_n):
            # Base index for this batch and channel
            base_idx = m * C * H * W + n * H * W
            
            for h_out in range(H_out):
                for w_out in range(W_out):
                    # Compute average for the pooling window
                    sum_val = 0.0
                    count = 0
                    
                    # Pool window: 2x2 with stride=2
                    for k_h in range(pool_size):
                        for k_w in range(pool_size):
                            h_in = h_out * stride + k_h
                            w_in = w_out * stride + k_w
                            
                            if 0 <= h_in and h_in < H:
                                if 0 <= w_in and w_in < W:
                                    x_offset = base_idx + h_in * W + w_in
                                    x_val = tl.load(x_ptr + x_offset, other=0.0)
                                    sum_val += x_val
                                    count += 1
                    
                    # Compute average
                    if count > 0:
                        avg_val = sum_val / count
                    else:
                        avg_val = 0.0
                    
                    # Store result
                    out_idx = m * C * H_out * W_out + n * H_out * W_out + h_out * W_out + w_out
                    tl.store(out_ptr + out_idx, avg_val)

@torch.fx.wrap
def optimized_avg_pool2d(x):
    # Get tensor shape
    N, C, H, W = x.shape
    H_out = H // 2  # stride=2
    W_out = W // 2  # stride=2
    
    # Create output tensor
    out = torch.empty((N, C, H_out, W_out), dtype=torch.float32, device=x.device)
    
    # Launch parameters
    BLOCK_SIZE_M = 1  # Process one batch element per program
    BLOCK_SIZE_N = 32  # Process 32 channels per program
    
    # Grid size
    grid_x = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_y = (C + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_z = 1
    
    # Launch kernel
    avg_pool2d_kernel[(grid_x, grid_y, grid_z)](
        x, out,
        N, C, H, W,
        2, 2,  # pool_size=2, stride=2
        BLOCK_SIZE_M, BLOCK_SIZE_N,
    )
    
    return out

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_avg_pool2d