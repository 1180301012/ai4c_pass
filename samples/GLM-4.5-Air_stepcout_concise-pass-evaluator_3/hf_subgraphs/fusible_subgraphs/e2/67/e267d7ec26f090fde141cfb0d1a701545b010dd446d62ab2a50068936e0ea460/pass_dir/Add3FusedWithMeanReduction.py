import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern matching for 3-input addition followed by mean reduction"""
    tmp_0 = in_1 + in_2
    tmp_0 += in_0
    tmp_1 = tmp_0
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_add3_mean_kernel(
    x_ptr, y_ptr, z_ptr,
    out_ptr, mean_ptr,
    n_channels, height, width,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """Fused kernel for 3-input addition + mean reduction"""
    pid_c = tl.program_id(0)  # channel dimension
    pid_n = tl.program_id(1)  # batch dimension
    
    # Calculate mean using shared memory and atomic operations
    block_sum = tl.zeros([BLOCK_SIZE_C], dtype=tl.float32)
    
    # Compute mean for this channel across spatial dimensions
    for h in range(0, height, BLOCK_SIZE_N):
        for w in range(0, width, BLOCK_SIZE_N):
            # Calculate spatial block boundaries
            h_end = min(h + BLOCK_SIZE_N, height)
            w_end = min(w + BLOCK_SIZE_N, width)
            
            for c in range(pid_c * BLOCK_SIZE_C, min((pid_c + 1) * BLOCK_SIZE_C, n_channels)):
                for hi in range(h, h_end):
                    for wi in range(w, w_end):
                        # Load inputs and add
                        x = tl.load(x_ptr + pid_n * n_channels * height * width + c * height * width + hi * width + wi)
                        y = tl.load(y_ptr + pid_n * n_channels * height * width + c * height * width + hi * width + wi)
                        z = tl.load(z_ptr + pid_n * n_channels * height * width + c * height * width + hi * width + wi)
                        sum_val = x + y + z
                        block_sum[0] += sum_val
    
    # Reduce across the block
    block_sum = tl.sum(block_sum, 0)
    
    # Store the mean result
    spatial_elements = height * width
    mean_val = block_sum / spatial_elements
    tl.store(mean_ptr + pid_n * n_channels + pid_c, mean_val)

@torch.fx.wrap
def fused_add3_mean(in_0, in_1, in_2):
    """Wrapper function for the fused kernel"""
    batch_size, n_channels, height, width = in_0.shape
    
    # Allocate output tensors
    out = torch.empty_like(in_0)
    mean = torch.empty(batch_size, n_channels, 1, 1, device=in_0.device, dtype=in_0.dtype)
    
    # Flatten for easier kernel access
    in_0_flat = in_0.reshape(-1)
    in_1_flat = in_1.reshape(-1)
    in_2_flat = in_2.reshape(-1)
    out_flat = out.reshape(-1)
    mean_flat = mean.reshape(-1)
    
    # Set up kernel launch parameters
    BLOCK_SIZE_C = 256
    BLOCK_SIZE_N = 16
    
    n_channel_blocks = (n_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    n_batch_blocks = batch_size
    
    # Launch kernel
    fused_add3_mean_kernel[(n_channel_blocks, n_batch_blocks)](
        in_0_flat, in_1_flat, in_2_flat,
        out_flat, mean_flat,
        n_channels, height, width,
        BLOCK_SIZE_N, BLOCK_SIZE_C
    )
    
    return out, mean

def replacement_func():
    return fused_add3_mean