import torch
import triton
import triton.language as tl

# Pattern matching function - match just the cat operation
def pattern(tmp_0, tmp_1, tmp_2, tmp_3):
    """
    Match just the cat operation - concatenating 4 tensors along dim 1
    """
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(tmp_0, tmp_1, tmp_2, tmp_3):
    return (tmp_0, tmp_1, tmp_2, tmp_3)


# Triton kernel for relu
@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, x, mask=mask)


def triton_relu(x):
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    relu_kernel[(num_programs,)](x, output, n_elements, BLOCK_SIZE)
    return output


# Optimized Triton kernel for max_pool2d with 5x5 kernel, stride=1, padding=2, dilation=1
@triton.jit
def max_pool2d_5x5_kernel(
    input_ptr, output_ptr,
    Batch: tl.constexpr, Channels: tl.constexpr,
    Height: tl.constexpr, Width: tl.constexpr,
    OutHeight: tl.constexpr, OutWidth: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized max_pool2d kernel with 5x5 kernel, stride=1, padding=2, dilation=1
    Output size equals input size with these parameters
    """
    # Each program processes one channel in one batch
    batch_idx = tl.program_id(0) // Channels
    channel_idx = tl.program_id(0) % Channels
    
    # Each thread block handles BLOCK_SIZE elements in output
    row_start = tl.program_id(1) * BLOCK_SIZE
    col_start = tl.program_id(2) * BLOCK_SIZE
    
    # Iterate over rows and cols
    offs_row = row_start + tl.arange(0, BLOCK_SIZE)
    offs_col = col_start + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for boundary checks
    row_mask = offs_row < OutHeight
    col_mask = offs_col < OutWidth
    
    # Maximum value for initialization
    neg_inf = float("-inf")
    
    max_vals = tl.full((BLOCK_SIZE, BLOCK_SIZE), neg_inf, tl.float32)
    
    # 5x5 kernel loop
    for kh in range(5):
        for kw in range(5):
            # Input position with padding (pad=2)
            in_row = offs_row + kh - 2
            in_col = offs_col + kw - 2
            
            # Check if within input bounds
            in_bounds = (in_row >= 0) & (in_row < Height) & (in_col >= 0) & (in_col < Width)
            
            # Calculate flat index for input
            in_offset = (batch_idx * Channels * Height * Width + 
                        channel_idx * Height * Width + 
                        in_row * Width + in_col)
            
            # Load with mask
            vals = tl.load(input_ptr + in_offset, mask=in_bounds, other=neg_inf)
            
            # Update max
            max_vals = tl.max(max_vals, vals)
    
    # Store result
    out_offset = (batch_idx * Channels * OutHeight * OutWidth +
                  channel_idx * OutHeight * OutWidth +
                  offs_row * OutWidth + offs_col)
    
    tl.store(output_ptr + out_offset, max_vals, mask=row_mask & col_mask)


def triton_max_pool2d_5x5(input_tensor):
    """
    Triton implementation of max_pool2d with 5x5 kernel, stride=1, padding=2, dilation=1
    """
    B, C, H, W = input_tensor.shape
    OutH = H
    OutW = W
    
    output = torch.empty((B, C, OutH, OutW), device=input_tensor.device, dtype=input_tensor.dtype)
    
    BLOCK_SIZE = 16
    grid = (B * C, (OutH + BLOCK_SIZE - 1) // BLOCK_SIZE, (OutW + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    max_pool2d_5x5_kernel[grid](
        input_tensor, output,
        B, C, H, W, OutH, OutW,
        BLOCK_SIZE
    )
    
    return output


# Triton kernel for concatenating 4 tensors along channel dimension
@triton.jit
def cat_4channel_kernel(
    input0_ptr, input1_ptr, input2_ptr, input3_ptr, output_ptr,
    Batch: tl.constexpr, Channels: tl.constexpr,
    Height: tl.constexpr, Width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Concatenate 4 tensors along channel dimension (dim=1)
    input shape: (B, C, H, W)
    output shape: (B, 4*C, H, W)
    """
    batch_idx = tl.program_id(0) // (Channels * 4)
    rem = tl.program_id(0) % (Channels * 4)
    c_out = rem // Channels
    c_in = rem % Channels
    
    row_start = tl.program_id(1) * BLOCK_SIZE
    col_start = tl.program_id(2) * BLOCK_SIZE
    
    offs_row = row_start + tl.arange(0, BLOCK_SIZE)
    offs_col = col_start + tl.arange(0, BLOCK_SIZE)
    
    row_mask = offs_row < Height
    col_mask = offs_col < Width
    
    # Select input based on output channel group
    input_ptr = tl.where(c_out == 0, input0_ptr,
                tl.where(c_out == 1, input1_ptr,
                tl.where(c_out == 2, input2_ptr, input3_ptr)))
    
    # Input channel index
    in_offset = (batch_idx * Channels * Height * Width + 
                c_in * Height * Width + 
                offs_row * Width + offs_col)
    
    # Output offset
    out_offset = (batch_idx * (Channels * 4) * Height * Width +
                  c_out * Channels * Height * Width +
                  c_in * Height * Width +
                  offs_row * Width + offs_col)
    
    vals = tl.load(input_ptr + in_offset, mask=row_mask & col_mask, other=0.0)
    tl.store(output_ptr + out_offset, vals, mask=row_mask & col_mask)


def triton_cat_4channel(t0, t1, t2, t3):
    """
    Concatenate 4 tensors along channel dimension using Triton
    """
    B, C, H, W = t0.shape
    output = torch.empty((B, C * 4, H, W), device=t0.device, dtype=t0.dtype)
    
    BLOCK_SIZE = 16
    grid = (B * C * 4, (H + BLOCK_SIZE - 1) // BLOCK_SIZE, (W + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    cat_4channel_kernel[grid](
        t0, t1, t2, t3, output,
        B, C, H, W,
        BLOCK_SIZE
    )
    
    return output


def optimized_function(tmp_0, tmp_1, tmp_2, tmp_3):
    """
    Optimized cat using Triton kernel
    """
    return triton_cat_4channel(tmp_0, tmp_1, tmp_2, tmp_3)


@torch.fx.wrap
def kernel_wrapper(tmp_0, tmp_1, tmp_2, tmp_3):
    return optimized_function(tmp_0, tmp_1, tmp_2, tmp_3)


def replacement_func():
    return kernel_wrapper