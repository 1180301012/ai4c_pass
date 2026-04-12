import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matching: ReLU + Addition + Adaptive Average Pooling with output size 1
    """
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_relu_add_mean_pool_kernel(
    x_ptr,           # in_0 pointer
    y_ptr,           # in_1 pointer  
    out_ptr,         # output pointer
    n_batch,         # batch size
    n_channels,      # number of channels
    height,          # input height
    width,           # input width
    BLOCK_SIZE_M: tl.constexpr,  # block size for batch dimension
    BLOCK_SIZE_N: tl.constexpr,  # block size for channel dimension
    BLOCK_SIZE_K: tl.constexpr   # block size for spatial dimension
):
    # Program identifiers
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # channel dimension
    
    # Compute ranges
    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for bounds checking
    m_mask = m_range < n_batch
    n_mask = n_range < n_channels
    
    # Initialize output sum for each (batch, channel) pair
    # We'll compute the sum across spatial dimensions and divide by H*W later
    if m_mask[0] and n_mask[0]:
        spatial_size = height * width
        out_sum = tl.zeros([], dtype=tl.float32)
        
        # Loop through spatial dimensions and accumulate sum
        for k in range(0, spatial_size, BLOCK_SIZE_K):
            k_end = min(k + BLOCK_SIZE_K, spatial_size)
            k_range = tl.arange(k, k_end)
            
            # Compute spatial coordinates
            h_coords = k_range // width
            w_coords = k_range % width
            
            # Create spatial mask
            k_mask = k_range < spatial_size
            
            # Load inputs for all batch and channels at this spatial location
            # We need to load one element per (batch, channel, spatial) combination
            for m_idx, m_val in enumerate(m_range):
                if not m_mask[m_idx]:
                    continue
                for n_idx, n_val in enumerate(n_range):
                    if not n_mask[n_idx]:
                        continue
                    
                    # Compute linear index for (m, n, k)
                    linear_idx = (m_val * n_channels + n_val) * spatial_size + k_range
                    
                    # Load x and y values at this spatial location
                    x_val = tl.load(x_ptr + linear_idx, mask=k_mask, other=0.0)
                    y_val = tl.load(y_ptr + linear_idx, mask=k_mask, other=0.0)
                    
                    # Apply ReLU to y, then add to x, accumulate sum
                    relu_y = tl.maximum(y_val, 0.0)
                    add_result = x_val + relu_y
                    out_sum += tl.sum(add_result * k_mask)
        
        # Store the mean (sum divided by spatial size)
        # Each thread processes one (batch, channel) pair
        out_idx = m_range[0] * n_channels + n_range[0]
        mean_val = out_sum / spatial_size if spatial_size > 0 else 0.0
        tl.store(out_ptr + out_idx, mean_val)

@torch.fx.wrap
def fused_relu_add_mean_pool(x, y):
    """
    Fusion of ReLU + Addition + Adaptive Average Pooling with size 1
    """
    # Get input shapes
    batch_size, channels, height, width = x.shape
    
    # Output shape should be [batch_size, channels, 1, 1]
    output_size = batch_size * channels
    
    # Determine optimal block sizes based on tensor dimensions
    BLOCK_SIZE_M = min(64, batch_size)
    BLOCK_SIZE_N = min(256, channels)
    BLOCK_SIZE_K = min(1024, height * width)
    
    # Calculate grid dimensions
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (grid_m, grid_n)
    
    # Create output tensor
    out = torch.empty(output_size, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    fused_relu_add_mean_pool_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_batch=batch_size,
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # Reshape output to [batch_size, channels, 1, 1]
    return out.reshape(batch_size, channels, 1, 1)

def replacement_func():
    return fused_relu_add_mean_pool