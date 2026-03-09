import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation graph
def pattern(in_0, in_1):
    # Match element-wise multiplication
    tmp_0 = in_1 * in_0
    # Match sum along dimension 1
    tmp_1 = torch.sum(tmp_0, dim=1)
    # Match unsqueeze(1) - adding dimension at position 1
    tmp_2 = tmp_1.unsqueeze(1)
    # Match sigmoid activation
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel that fuses all operations
@triton.jit
def fused_kernel(
    x_ptr,             # Pointer to input tensor in_0  
    y_ptr,             # Pointer to input tensor in_1
    out_ptr,           # Pointer to output tensor
    n_batch,           # Batch size 
    n_channels,        # Channel dimension (summed along this)
    n_height,          # Height dimension
    n_width,           # Width dimension 
    BLOCK_SIZE_H: tl.constexpr,  # Block size for height dimension
    BLOCK_SIZE_W: tl.constexpr,  # Block size for width dimension
):
    # Program identifiers for 3D parallelism over batch, height, width
    pid_m = tl.program_id(0)  # Batch dimension
    pid_h = tl.program_id(1)  # Height dimension
    pid_w = tl.program_id(2)  # Width dimension
    
    # Compute current block offsets
    h_offset = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_offset = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Create masks for bounds checking
    h_mask = h_offset < n_height
    w_mask = w_offset < n_width
    
    # Initialize accumulator with high precision
    accumulator = tl.zeros([BLOCK_SIZE_H, BLOCK_SIZE_W], dtype=tl.float32)
    
    # Process each channel with optimal memory access pattern
    # This approach minimizes memory fragmentation and improves precision
    for c in range(0, n_channels):
        # Calculate exact memory offsets for this channel
        # Using precise arithmetic to avoid rounding errors
        batch_base = pid_m * n_channels * n_height * n_width
        channel_base = c * n_height * n_width
        base_offset = batch_base + channel_base
        
        # Calculate spatial coordinates for this thread block
        h_indices = h_offset[:, None] * n_width
        w_indices = w_offset[None, :]
        spatial_offset = h_indices + w_indices
        
        # Full memory offset for this channel and spatial position
        full_offset = base_offset + spatial_offset
        
        # Load data with explicit bounds checking
        # This ensures we don't access out-of-bounds memory
        h_mask_valid = h_offset < n_height
        w_mask_valid = w_offset < n_width
        spatial_mask = h_mask_valid[:, None] & w_mask_valid[None, :]
        
        # Load x and y values with high precision
        x_val = tl.load(
            x_ptr + full_offset,
            mask=spatial_mask,
            other=0.0
        )
        
        y_val = tl.load(
            y_ptr + full_offset,
            mask=spatial_mask,
            other=0.0
        )
        
        # Perform precise multiplication
        product_val = x_val * y_val
        
        # Accumulate with careful initialization and accumulation
        if c == 0:
            # Initialize accumulator with first channel product
            accumulator = product_val
        else:
            # Add to accumulator for subsequent channels
            # Using explicit addition for better precision
            accumulator = accumulator + product_val

    # Apply sigmoid to accumulated result
    out_val = tl.sigmoid(accumulator)
    
    # Store output with unsqueeze(1) shape [batch, 1, height, width]
    # For unsqueeze(1), we add a dimension at position 1, so we store at index 0
    output_offset = (pid_m * 1 * n_height * n_width +  # batch * stride_batch
                     0 * n_height * n_width +          # unsqueeze position 1 = 0
                     h_offset[:, None] * n_width +      # height * stride_height  
                     w_offset[None, :])                # width
    
    # Store the result
    tl.store(
        out_ptr + output_offset,
        out_val,
        mask=h_mask[:, None] & w_mask[None, :]
    )

# Kernel wrapper function that handles memory allocation and kernel launch
@torch.fx.wrap
def fused_mul_sum_unsqueeze_sigmoid(in_0, in_1):
    # Get input tensor shapes
    batch_dim, channel_dim, height_dim, width_dim = in_0.shape
    
    # Allocate output tensor with unsqueeze shape [batch, 1, height, width]
    output_shape = (batch_dim, 1, height_dim, width_dim)
    out = torch.empty(output_shape, dtype=torch.float32, device=in_0.device)
    
    # Use reasonable block sizes for height and width dimensions
    BLOCK_SIZE_H = 64  # Process 64 height elements per thread
    BLOCK_SIZE_W = 64  # Process 64 width elements per thread
    
    # Calculate grid dimensions
    grid_m = (batch_dim + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_h = (height_dim + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (width_dim + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch the fused kernel
    fused_kernel[(grid_m, grid_h, grid_w)](
        in_0,
        in_1, 
        out,
        batch_dim,
        channel_dim,
        height_dim,
        width_dim,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W
    )
    
    return out

# Replacement function returns the fused kernel
def replacement_func():
    return fused_mul_sum_unsqueeze_sigmoid