import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match the addition operation at the beginning of the computation"""
    return in_1 + in_0

def replacement_args(in_0, in_1):
    """Extract input tensors for the addition operation"""
    return (in_0, in_1)

@triton.jit
def triton_softmax_kernel(x_ptr, out_ptr, n_cols, n_rows, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """
    High-performance softmax kernel using Triton
    Optimized for the specific shape patterns in the target computation
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create row offsets 
    row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = row_offsets < n_rows
    
    # Create column offsets
    col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = col_offsets < n_cols
    
    # Load data - each program loads a tile
    x = tl.load(x_ptr + row_offsets[:, None] * n_cols + col_offsets[None, :], 
                mask=mask_m[:, None] & mask_n[None, :], 
                other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(x, axis=1)
    row_max = tl.where(mask_m, row_max, -float('inf'))
    
    # Subtract max and compute exp
    x_exp = tl.exp(x - row_max[:, None])
    
    # Compute sum per row
    row_sum = tl.sum(x_exp, axis=1)
    row_sum = tl.where(mask_m, row_sum, 1.0)  # Avoid division by zero
    
    # Normalize to get softmax
    softmax_result = x_exp / row_sum[:, None]
    
    # Store result
    tl.store(out_ptr + row_offsets[:, None] * n_cols + col_offsets[None, :], 
             softmax_result, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def triton_add_kernel_broadcast(x_ptr, y_ptr, out_ptr, n_elements, 
                                x_batch: tl.constexpr, x_channels: tl.constexpr,
                                y_batch: tl.constexpr, y_channels: tl.constexpr, 
                                height: tl.constexpr, width: tl.constexpr,
                                BLOCK_SIZE: tl.constexpr):
    """Broadcasting addition kernel that expands [1, 1, H, W] to [1, 8, H, W]"""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate indices for the 4D tensors using passed constants
    # y has shape [1, 8, H, W], x has shape [1, 1, H, W]
    total_hw = height * width
    hw_idx = offsets % total_hw
    w_idx = hw_idx % width
    h_idx = hw_idx // width
    batch_idx = 0  # first batch dimension is always 0
    channel_idx = offsets // total_hw  # second dimension (0-7 for 8 channels)
    
    # For x, we always use channel_idx = 0 (since it only has one channel)
    x_hw_offset = (batch_idx * x_channels + 0) * total_hw + h_idx * width + w_idx
    x_offset = x_hw_offset
    
    # Broadcast x by using the same value for all channels
    # Load x data (only from channel 0) and y data
    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def triton_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """High-performance addition kernel using Triton"""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(in_0, in_1):
    """High-performance addition using Triton kernel with broadcasting support"""
    # Handle broadcasting from [1, 1, H, W] to [1, 8, H, W]
    # Create output tensor with shape of in_1
    out = torch.empty_like(in_1)
    
    if len(in_0.shape) == 4 and len(in_1.shape) == 4 and in_0.shape[1] == 1 and in_1.shape[1] == 8:
        # Broadcast in_0 from [1, 1, H, W] to [1, 8, H, W] before adding
        n_elements = in_1.numel()
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Extract tensor dimensions
        x_batch, x_channels = in_0.shape[0], in_0.shape[1]
        y_batch, y_channels = in_1.shape[0], in_1.shape[1]
        height, width = in_0.shape[2], in_0.shape[3]  # H and W should be same
        
        triton_add_kernel_broadcast[(num_programs,)](
            x_ptr=in_0,
            y_ptr=in_1, 
            out_ptr=out,
            n_elements=n_elements,
            x_batch=x_batch,
            x_channels=x_channels,
            y_batch=y_batch,
            y_channels=y_channels,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Fallback to simple addition for non-broadcasting cases
        n_elements = in_0.numel()
        BLOCK_SIZE = 1024  
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        triton_add_kernel[(num_programs,)](
            x_ptr=in_0,
            y_ptr=in_1,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return out

def replacement_func():
    return triton_add