import torch
import triton
import triton.language as tl
import math

def pattern(x):
    tmp_0 = x * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def fused_kernel(x_ptr, out_ptr, batch_size, num_features, height, width, MAX_WIDTH: tl.constexpr):
    # Each program handles one (batch, feature, height) and processes all width positions
    pid_batch = tl.program_id(0)  # batch
    pid_feat = tl.program_id(1)  # feature
    pid_h = tl.program_id(2)    # height
    
    # Check bounds
    if pid_batch >= batch_size or pid_feat >= num_features or pid_h >= height:
        return
    
    # Fixed width offsets (MAX_WIDTH elements)
    width_offsets = tl.arange(0, MAX_WIDTH)
    
    # Create mask for valid width indices (actual width might be smaller than MAX_WIDTH)
    width_mask = width_offsets < width
    
    # Load all width positions for this row (masked to actual width size)
    row_data = tl.load(
        x_ptr + pid_batch * num_features * height * width + 
        pid_feat * height * width + pid_h * width + width_offsets,
        mask=width_mask,
        other=0.0
    ).to(tl.float32)
    
    # Apply scalar multiplication
    scaled_data = row_data * 0.1767766952966369
    
    # Compute softmax along the width dimension
    max_val = tl.max(scaled_data)
    exp_data = tl.exp(scaled_data - max_val)
    sum_exp = tl.sum(exp_data)
    softmax_data = exp_data / sum_exp
    
    # Store the softmax data in transposed positions
    # For each width position, store it at the transposed location
    # Original: (batch, feat, height, width) -> Output: (batch, feat, width, height)
    # So for our specific (batch, feat, height), we store at (batch, feat, width, height)
    
    out_offset = pid_batch * num_features * height * width + \
                pid_feat * height * width + \
                width_offsets * height + pid_h
    
    tl.store(
        out_ptr + out_offset,
        softmax_data,
        mask=width_mask
    )




@torch.fx.wrap
def fused_kernel_wrapper(x):
    shape = x.shape  # [batch_size, num_features, height, width]
    batch_size, num_features, height, width = shape
    
    # Set maximum width (should be larger than any input width)
    MAX_WIDTH = 512  # Safe upper bound for typical spatial dimensions
    
    # Calculate grid dimensions for 3D grid: (batch, features, height)
    grid_batch = batch_size
    grid_features = num_features
    grid_height = height
    
    # Create output tensor
    out = torch.empty_like(x, dtype=torch.float32)
    
    # Launch the optimized kernel that does all three operations:
    # 1. Scalar multiplication
    # 2. Softmax along last dimension 
    # 3. Transpose of last two dimensions
    fused_kernel[(grid_batch, grid_features, grid_height)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        num_features=num_features,
        height=height,
        width=width,
        MAX_WIDTH=MAX_WIDTH
    )
    
    return out

def replacement_func():
    return fused_kernel_wrapper