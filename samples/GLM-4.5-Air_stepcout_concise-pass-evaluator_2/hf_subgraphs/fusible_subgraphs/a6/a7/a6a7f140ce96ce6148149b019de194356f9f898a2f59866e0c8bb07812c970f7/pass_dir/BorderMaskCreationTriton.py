import torch
import triton
import triton.language as tl
import math

@triton.jit
def border_mask_kernel_h_w(
    out_ptr,
    H, W,
    border_size: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """Create border masks more efficiently with Triton"""
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Compute offsets for this program
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    
    # Create offsets within the block
    offsets_h = h_start + tl.arange(0, BLOCK_SIZE_H)
    offsets_w = w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create mask for valid indices
    mask_h = offsets_h < H
    mask_w = offsets_w < W
    mask = mask_h[:, None] & mask_w[None, :]
    
    # Initialize output to 0
    out = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Set border regions to 1
    # Bottom border
    bottom_mask = offsets_h >= (H - border_size)
    out = tl.where(bottom_mask[:, None] & mask_w[None, :], 1.0, out)
    
    # Right border  
    right_mask = offsets_w >= (W - border_size)
    out = tl.where(mask_h[:, None] & right_mask[None, :], 1.0, out)
    
    # Corner overlap is handled automatically (it will be 1)
    
    # Store result
    out_ptrs = out_ptr + offsets_h[:, None] * W + offsets_w[None, :]
    tl.store(out_ptrs, out, mask=mask)

@torch.fx.wrap
def create_border_masks_triton(H, W, border_size=5):
    """Create both row and column border masks efficiently"""
    # The device will be inferred from the context, don't explicitly create device
    
    # Create output tensors
    row_mask = torch.zeros((1, H, W), dtype=torch.float32)
    col_mask = torch.zeros((1, H, W), dtype=torch.float32)
    
    # Calculate grid sizes
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    
    # For row mask (bottom border)
    grid_h_row = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w_row = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    border_mask_kernel_h_w[(grid_h_row, grid_w_row)](
        row_mask,
        H, W,
        border_size,
        BLOCK_SIZE_H, BLOCK_SIZE_W
    )
    
    # For column mask (right border) 
    grid_h_col = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w_col = (W + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    border_mask_kernel_h_w[(grid_h_col, grid_w_col)](
        col_mask,
        H, W,
        border_size,
        BLOCK_SIZE_H, BLOCK_SIZE_W
    )
    
    return row_mask, col_mask

def pattern():
    """Pattern matching the original mask creation operations"""
    # This creates the original pattern that needs to be matched
    # We need to match the zeros creation and the specific slice/fill operations
    device = torch.device(type='cuda', index=0)
    tmp_0 = torch.zeros((1, 133, 133), device=device)
    tmp_1 = tmp_0[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
    tmp_2 = tmp_1.fill_(1.0)
    return tmp_2

def replacement_args():
    # Extract arguments needed for replacement
    # In this case, we just need the shape and border size
    return (133, 133, 5)

def replacement_func():
    """Return the optimized border mask creation function"""
    return create_border_masks_triton