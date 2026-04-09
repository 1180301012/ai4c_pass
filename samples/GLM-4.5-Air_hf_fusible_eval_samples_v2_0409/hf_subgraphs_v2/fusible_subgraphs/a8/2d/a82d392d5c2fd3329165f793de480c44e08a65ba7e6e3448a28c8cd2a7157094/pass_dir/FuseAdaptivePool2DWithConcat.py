import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the model.py computation
def pattern(in_0, in_1):
    """
    Matches adaptive_avg_pool2d followed by concatenation with another tensor
    along the channel dimension (dim=1)
    """
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Helper kernels for the computation
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr,
    y_ptr,
    n_batch,
    channels,
    in_height,
    in_width,
    target_height,
    target_width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Simple adaptive average pooling kernel for 2D
    """
    pid_m = tl.program_id(0)  # batch
    pid_n = tl.program_id(1)  # channel
    
    if pid_m >= n_batch or pid_n >= channels:
        return
    
    # Each thread handles one output position
    for sp_idx in range(target_height * target_width):
        h_out = sp_idx // target_width
        w_out = sp_idx % target_width
        
        # Calculate source region
        h_in_start = (h_out * in_height) // target_height
        h_in_end = ((h_out + 1) * in_height) // target_height
        w_in_start = (w_out * in_width) // target_width
        w_in_end = ((w_out + 1) * in_width) // target_width
        
        # Clamp to valid range
        h_in_end = min(h_in_end, in_height)
        w_in_end = min(w_in_end, in_width)
        
        # Compute average
        sum_val = 0.0
        count = 0
        
        stride_x = in_height * in_width
        src_base = (pid_m * channels + pid_n) * stride_x
        
        for h_in in range(h_in_start, h_in_end):
            for w_in in range(w_in_start, w_in_end):
                ptr = src_base + h_in * in_width + w_in
                val = tl.load(x_ptr + ptr)
                sum_val += val
                count += 1
        
        if count > 0:
            avg_val = sum_val / count
            stride_y = target_height * target_width
            out_ptr_base = (pid_m * channels + pid_n) * stride_y + h_out * target_width + w_out
            tl.store(y_ptr + out_ptr_base, avg_val)

@triton.jit  
def concat_kernel(
    pooled_ptr,
    in_1_ptr,
    out_ptr,
    n_batch,
    pooled_channels,
    in_1_channels,
    target_height,
    target_width,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Concatenation kernel that appends in_1 after pooled result
    """
    pid_m = tl.program_id(0)  # batch
    pid_n = tl.program_id(1)  # channel block
    
    if pid_m >= n_batch:
        return
    
    # Calculate channel range for this block
    channel_start = pid_n * BLOCK_SIZE_N
    out_channels = pooled_channels + in_1_channels
    channel_end = min(channel_start + BLOCK_SIZE_N, out_channels)
    
    # Process each channel in this block
    for c_out in range(channel_start, channel_end):
        stride_in = target_height * target_width
        
        if c_out < pooled_channels:
            # Copy from pooled result
            src_base = (pid_m * pooled_channels + c_out) * stride_in
            for sp_idx in range(stride_in):
                out_ptr_base = (pid_m * out_channels + c_out) * stride_in + sp_idx
                val = tl.load(pooled_ptr + src_base + sp_idx)
                tl.store(out_ptr + out_ptr_base, val)
        else:
            # Copy from in_1
            c_in_1 = c_out - pooled_channels
            src_base = (pid_m * in_1_channels + c_in_1) * stride_in
            for sp_idx in range(stride_in):
                out_ptr_base = (pid_m * out_channels + c_out) * stride_in + sp_idx
                val = tl.load(in_1_ptr + src_base + sp_idx)
                tl.store(out_ptr + out_ptr_base, val)

# Kernel wrapper function
@torch.fx.wrap
def fused_adaptive_pool_concat(in_0, in_1):
    """
    Performs fused adaptive average pooling and concatenation
    """
    # Get tensor shapes  
    n_batch = in_0.shape[0]
    in_0_channels = in_0.shape[1]
    in_0_height = in_0.shape[2]
    in_0_width = in_0.shape[3]
    
    in_1_channels = in_1.shape[1]
    in_1_height = in_1.shape[2]
    in_1_width = in_1.shape[3]
    
    # Output dimensions
    target_height = 32
    target_width = 24
    out_channels = in_0_channels + in_1_channels
    
    # Step 1: Perform adaptive average pooling on in_0
    pooled_shape = [n_batch, in_0_channels, target_height, target_width]
    pooled = torch.empty(pooled_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Set up grid for pooling kernel
    BLOCK_SIZE_M_POOL = 1  # batch dimension  
    BLOCK_SIZE_N_POOL = 64  # channels per program
    grid_m_pool = n_batch
    grid_n_pool = (in_0_channels + BLOCK_SIZE_N_POOL - 1) // BLOCK_SIZE_N_POOL
    
    # Launch pooling kernel
    adaptive_avg_pool2d_kernel[(grid_m_pool, grid_n_pool)](
        x_ptr=in_0,
        y_ptr=pooled,
        n_batch=n_batch,
        channels=in_0_channels,
        in_height=in_0_height,
        in_width=in_0_width,
        target_height=target_height,
        target_width=target_width,
        BLOCK_SIZE_M=BLOCK_SIZE_M_POOL,
        BLOCK_SIZE_N=BLOCK_SIZE_N_POOL,
    )
    
    # Step 2: Concatenate pooled result with in_1
    output_shape = [n_batch, out_channels, target_height, target_width]
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Set up grid for concatenation kernel
    BLOCK_SIZE_N_CONCAT = 64  # channel blocks per program
    grid_m_concat = n_batch
    grid_n_concat = (out_channels + BLOCK_SIZE_N_CONCAT - 1) // BLOCK_SIZE_N_CONCAT
    
    # Launch concatenation kernel
    concat_kernel[(grid_m_concat, grid_n_concat)](
        pooled_ptr=pooled,
        in_1_ptr=in_1,
        out_ptr=out,
        n_batch=n_batch,
        pooled_channels=in_0_channels,
        in_1_channels=in_1_channels,
        target_height=target_height,
        target_width=target_width,
        BLOCK_SIZE_N=BLOCK_SIZE_N_CONCAT,
    )
    
    return out

# Replacement function - returns the optimization function
def replacement_func():
    return fused_adaptive_pool_concat