import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Match the complete computation: relu + scale * + bias + padding"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = tmp_1 * tmp_2
    tmp_1 = tmp_2 = None
    tmp_4 = tmp_3 + tmp_0
    tmp_3 = tmp_0 = None
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    tmp_4 = None
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def full_comp_fused_kernel(
    bias_ptr,              # bias tensor [1]
    scale_ptr,             # scale tensor [1] 
    x_ptr,                 # input feature map [N, C, H, W]
    out_ptr,               # output [N, C, H+1, W+1]
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for complete computation: relu + scale * + bias + padding"""
    # Program handles one element from input
    pid = tl.program_id(0)
    
    # Total input elements (no padding)
    input_total = N * C * H * W
    
    # Calculate offsets within this program
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    # Input mask (only valid input elements)
    input_mask = offsets < input_total
    
    # Load bias and scale (scalar broadcast)
    bias = tl.load(bias_ptr)
    scale = tl.load(scale_ptr)
    
    # Load input feature map values
    x = tl.load(x_ptr + offsets, mask=input_mask, other=0.0)
    
    # Step 1: ReLU
    relu_x = tl.maximum(x, 0.0)
    
    # Step 2: Scale multiplication
    scale_relu = scale * relu_x
    
    # Step 3: Add bias
    bias_added = scale_relu + bias
    
    # Step 4: Store result (same position for existing elements)
    tl.store(out_ptr + offsets, bias_added, mask=input_mask)
    
    # Step 5: Handle padding - pad right and bottom edges with 0
    # Check if we're near the right or bottom edges
    for i in range(BLOCK_SIZE):
        elem_pos = offset + i
        if elem_pos >= input_total:
            break
            
        if input_mask[i]:
            # Calculate element coordinates
            elem_coord = elem_pos % (C * H * W)
            c = elem_coord // (H * W)
            h = elem_coord // W
            w = elem_coord % W
            
            # Handle right edge padding (pad 0 on the right)
            if w == W - 1:
                pad_right_pos = elem_pos + 1  # Right padding
                if pad_right_pos < input_total + W + 1:  # Within output bounds
                    tl.store(out_ptr + pad_right_pos, 0.0)
                    
                # Handle bottom edge padding
                if h == H - 1:  # Also bottom edge
                    pad_bottom_left = elem_pos + W  # Bottom position (left)
                    pad_bottom_right = elem_pos + W + 1  # Bottom position (right)
                    
                    if pad_bottom_left < input_total + W + 1:
                        tl.store(out_ptr + pad_bottom_left, bias_added[i])
                    if pad_bottom_right < input_total + W + 1:
                        tl.store(out_ptr + pad_bottom_right, 0.0)
            elif h == H - 1:  # Bottom edge but not right edge
                pad_bottom = elem_pos + W  # Direct bottom padding
                if pad_bottom < input_total + W + 1:
                    tl.store(out_ptr + pad_bottom, 0.0)

@torch.fx.wrap
def full_computation_triton(bias, scale, x):
    """PyTorch wrapper for the complete computation kernel"""
    if x.dim() != 4:
        # Fallback for non-4D tensors using original computation
        tmp_2 = torch.nn.functional.relu(x, inplace=False)
        tmp_3 = scale * tmp_2
        tmp_4 = tmp_3 + bias
        return torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    
    N, C, H, W = x.shape
    
    # Calculate output shape with padding  
    out_shape = (N, C, H + 1, W + 1)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Set block size based on tensor size
    if x.numel() < 16384:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 1024
        
    # Calculate grid dimensions
    total_elements = N * C * H * W
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_programs,)
    
    # Launch fused kernel
    full_comp_fused_kernel[grid](
        bias_ptr=bias,
        scale_ptr=scale,
        x_ptr=x,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    """Return the fused complete computation function"""
    return full_computation_triton