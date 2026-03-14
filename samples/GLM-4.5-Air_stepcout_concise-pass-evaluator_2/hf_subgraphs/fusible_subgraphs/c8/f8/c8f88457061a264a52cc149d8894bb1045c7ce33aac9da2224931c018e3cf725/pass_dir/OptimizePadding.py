import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    """Match the padding operation with specific padding pattern"""
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5

def replacement_args(tmp_4):
    """Extract arguments for the padding kernel"""
    return (tmp_4,)

@triton.jit
def optimized_pad_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for padding with (0,1,0,1) pattern"""
    # Each program handles one element
    pid = tl.program_id(0)
    
    # Total elements in input (no padding)
    input_total = N * C * H * W
    
    # Calculate offsets
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    # Input mask (only valid input elements)
    input_mask = offsets < input_total
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=input_mask, other=0.0)
    
    # Store output - same position for existing elements
    tl.store(out_ptr + offsets, x, mask=input_mask)
    
    # Handle padding elements for the bottom-right pixel
    # We need to pad on the right (last dim) and bottom (3rd last dim)
    if offset + BLOCK_SIZE - 1 >= input_total - W - 1:  # Near boundary
        # Check if we need to pad right edge
        for i in range(BLOCK_SIZE):
            elem_pos = offset + i
            if elem_pos >= input_total:
                break
                
            elem_coord = elem_pos % (C * H * W)
            c = elem_coord // (H * W)
            elem_coord = elem_coord % (H * W)
            h = elem_coord // W
            w = elem_coord % W
            
            if w == W - 1:  # Right edge
                # Pad right position
                pad_right_pos = elem_pos + 1
                tl.store(out_ptr + pad_right_pos, x[i], mask=input_mask[i:i+1])
            
            if h == H - 1 and w == W - 1:  # Bottom-right corner
                # Pad bottom-right position
                pad_bottom_right = elem_pos + W + 1
                tl.store(out_ptr + pad_bottom_right, x[i], mask=input_mask[i:i+1])
                
                # Pad bottom-left position  
                pad_bottom_left = elem_pos + W
                if pad_bottom_left < input_total + W + 1:  # Within output bounds
                    tl.store(out_ptr + pad_bottom_left, x[i], mask=input_mask[i:i+1])

@triton.jit
def efficient_pad_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """More efficient padding kernel for (0,1,0,1) pattern"""
    # 2D program grid for better memory locality
    batch = tl.program_id(0)
    channel = tl.program_id(1)
    
    # Calculate base offset for this batch/channel
    base_offset = batch * C * H * W + channel * H * W
    
    # Handle one channel per program
    for h in range(H):
        for w in range(W):
            # Check if this pixel needs padding
            element_offset = base_offset + h * W + w
            
            if w < W - 1:  # Not right edge
                # Normal pixel -> copy directly
                pixel_val = tl.load(x_ptr + element_offset)
                tl.store(out_ptr + element_offset, pixel_val)
            else:  # Right edge
                # Copy the pixel
                pixel_val = tl.load(x_ptr + element_offset)
                tl.store(out_ptr + element_offset, pixel_val)
                
                # Pad one pixel to the right (constant=0)
                pad_right_offset = element_offset + 1
                tl.store(out_ptr + pad_right_offset, 0.0)
                
                if h < H - 1:  # Not bottom edge
                    # No bottom padding needed for this row
                    pass
                else:  # Bottom edge (also right edge)
                    # Pad bottom pixels for this column
                    pad_bottom_offset = element_offset + W
                    tl.store(out_ptr + pad_bottom_offset, pixel_val)
                    
                    # Pad bottom-right corner
                    pad_bottom_right_offset = element_offset + W + 1  
                    tl.store(out_ptr + pad_bottom_right_offset, 0.0)

@torch.fx.wrap 
def optimized_pad_triton(x):
    """PyTorch wrapper for the padding kernel"""
    if x.dim() != 4:
        # For non-4D tensors, we'll create a simple fallback using tensor operations
        # Pad only the last two dimensions
        pad_right = torch.zeros(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)
        pad_result = torch.cat([x, pad_right], dim=-1)
        pad_bottom = torch.zeros(pad_result.shape[:-2] + (1, pad_result.shape[-1]), dtype=x.dtype, device=x.device)
        return torch.cat([pad_result, pad_bottom], dim=-2)
    
    N, C, H, W = x.shape
    
    # Calculate output shape with padding
    out_shape = (N, C, H + 1, W + 1)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Choose kernel based on tensor size
    if x.numel() < 16384:  # Small tensors
        BLOCK_SIZE = 64
        grid = (N, C)  
        efficient_pad_kernel[grid](
            x_ptr=x,
            out_ptr=out,
            N=N, C=C, H=H, W=W,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:  # Large tensors
        BLOCK_SIZE = 1024
        total_elements = N * C * H * W
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (num_programs,)
        optimized_pad_kernel[grid](
            x_ptr=x,
            out_ptr=out,
            N=N, C=C, H=H, W=W,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return out

def replacement_func():
    """Return the padding function"""
    return optimized_pad_triton