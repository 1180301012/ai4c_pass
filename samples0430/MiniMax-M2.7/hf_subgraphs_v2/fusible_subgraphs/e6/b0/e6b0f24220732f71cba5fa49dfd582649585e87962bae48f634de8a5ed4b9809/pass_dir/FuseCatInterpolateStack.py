import torch
import triton
import triton.language as tl


def pattern(x, y):
    """
    Simple pattern: just match cat
    """
    return torch.cat((x, y), 1)


def replacement_args(x, y):
    return (x, y)


@triton.jit
def triton_cat_kernel(
    out_ptr, out_stride_1, out_stride_2, out_stride_3,
    x_ptr, x_stride_1, x_stride_2, x_stride_3,
    y_ptr, y_stride_1, y_stride_2, y_stride_3,
    B, C1, C2, H, W, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for concatenating x and y along dim=1
    Each program handles one batch element
    """
    pid_b = tl.program_id(0)
    rn = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Channel height width linear index
    chw_idx = rn
    mask = chw_idx < (C1 + C2) * H * W
    
    c_idx = chw_idx // (H * W)
    h_idx = (chw_idx % (H * W)) // W
    w_idx = chw_idx % W
    
    # Check if from x or y
    from_x = c_idx < C1
    
    # Offsets within the batch
    x_offset = c_idx * x_stride_1 + h_idx * x_stride_2 + w_idx * x_stride_3
    y_offset = (c_idx - C1) * y_stride_1 + h_idx * y_stride_2 + w_idx * y_stride_3
    out_offset = c_idx * out_stride_1 + h_idx * out_stride_2 + w_idx * out_stride_3
    
    # Load values with proper masking
    x_val = tl.load(x_ptr + x_offset, mask=mask & from_x, other=0.0)
    y_val = tl.load(y_ptr + y_offset, mask=mask & ~from_x, other=0.0)
    
    # Select based on which tensor
    val = tl.where(from_x, x_val, y_val)
    
    # Store to output
    tl.store(out_ptr + out_offset, val, mask=mask)


@torch.fx.wrap
def optimized_cat(x, y):
    """
    Optimized cat using Triton kernel
    """
    B, C1, H, W = x.shape
    B2, C2, H2, W2 = y.shape
    
    output = torch.empty((B, C1 + C2, H, W), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 128
    total = (C1 + C2) * H * W
    num_programs = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use 2D grid: (batch, elements_per_batch)
    grid = (B, num_programs)
    
    triton_cat_kernel[grid](
        output, output.stride(1), output.stride(2), output.stride(3),
        x, x.stride(1), x.stride(2), x.stride(3),
        y, y.stride(1), y.stride(2), y.stride(3),
        B, C1, C2, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return optimized_cat