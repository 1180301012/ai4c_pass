import torch
import triton
import triton.language as tl

def pattern(tmp_8):
    # AdaptiveAvgPool2d operation
    tmp_9 = torch.nn.functional.adaptive_avg_pool2d(tmp_8, 1)
    # Flatten operation
    tmp_10 = tmp_9.flatten(1, -1)
    return tmp_9, tmp_10

def replacement_args(tmp_8):
    return (tmp_8,)

@triton.jit
def fused_pool_flatten_kernel(
    input_ptr,      # Input tensor [N, C, H, W]
    output_ptr,     # Output tensor [N, C]
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Calculate indices
    n_start = pid_n * BLOCK_SIZE_N
    c_offset = pid_c * BLOCK_SIZE_C
    
    # Process one batch and channel block
    for n in range(n_start, min(n_start + BLOCK_SIZE_N, N)):
        for c in range(c_offset, min(c_offset + BLOCK_SIZE_C, C)):
            # Compute global average across H and W
            sum_val = 0.0
            for h in range(H):
                for w in range(W):
                    input_val = tl.load(input_ptr + n * C * H * W + c * H * W + h * W + w)
                    sum_val += input_val
            
            # Average by dividing by total elements (H * W)
            avg_val = sum_val / (H * W)
            
            # Store result at flattened position
            tl.store(output_ptr + n * C + c, avg_val)

@torch.fx.wrap  
def fused_adaptive_avg_pool_flatten(input_tensor):
    N, C, H, W = input_tensor.shape
    
    # For adaptive avg pool with output size 1, result is [N, C, 1, 1] which flattens to [N, C]
    
    # Calculate optimal block sizes
    BLOCK_SIZE_N = 32  # Process multiple batches together
    BLOCK_SIZE_C = 32  # Process multiple channels together
    
    # Calculate grid dimensions
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    output = torch.empty((N, C), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    fused_pool_flatten_kernel[(grid_n, grid_c)](
        input_ptr=input_tensor,
        output_ptr=output,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return output

def replacement_func():
    return fused_adaptive_avg_pool_flatten