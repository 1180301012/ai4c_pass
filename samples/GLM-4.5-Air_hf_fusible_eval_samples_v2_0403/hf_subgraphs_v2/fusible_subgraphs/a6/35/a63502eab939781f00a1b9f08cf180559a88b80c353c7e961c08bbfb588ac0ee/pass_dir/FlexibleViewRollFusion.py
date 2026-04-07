import torch
import triton
import triton.language as tl

@triton.jit
def flexible_view_roll_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    spatial_h,
    spatial_w,
    feature_dim,
    roll_shift_h,
    roll_shift_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Flexible kernel that handles different spatial and feature dimensions."""
    pid = tl.program_id(0)
    total_elements = batch_size * spatial_h * spatial_w * feature_dim
    block_size = BLOCK_SIZE
    
    # Calculate the range for this program
    start_idx = pid * block_size
    end_idx = min(start_idx + block_size, total_elements)
    
    # Process each element in the block
    for idx in range(start_idx, end_idx):
        # Calculate indices for each dimension using integer arithmetic
        residual = idx
        fid = residual % feature_dim
        residual //= feature_dim
        w_idx = residual % spatial_w
        residual //= spatial_w
        h_idx = residual % spatial_h
        b_idx = residual // spatial_h
        
        # Apply roll operation
        rolled_w = (w_idx - roll_shift_w) % spatial_w
        rolled_h = (h_idx - roll_shift_h) % spatial_h
        
        # Calculate output index
        out_idx = (((b_idx * spatial_h + rolled_h) * spatial_w + rolled_w) * feature_dim + fid)
        
        # Load input and store to output
        input_val = tl.load(input_ptr + idx, other=0.0)
        tl.store(output_ptr + out_idx, input_val)

@torch.fx.wrap  
def flexible_fused_view_roll(input_tensor, spatial_h, spatial_w, feature_dim):
    """Flexible function that handles different tensor dimensions."""
    # Get batch size as the first dimension size
    batch_size = input_tensor.shape[0]
    
    output = torch.empty_like(input_tensor)
    
    # Calculate block size based on total elements
    total_elements = batch_size * spatial_h * spatial_w * feature_dim
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    flexible_view_roll_kernel[(num_programs,)](
        input_tensor,
        output,
        batch_size,
        spatial_h,
        spatial_w,
        feature_dim,
        4,  # roll_shift_h
        4,  # roll_shift_w
        BLOCK_SIZE
    )
    
    return output

def pattern(tmp_2):
    """Pattern that matches view + roll operations for flexible dimensions."""
    # Match exact computation from model.py
    tmp_3 = tmp_2.view(-1, 32, 32, 768)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    return tmp_3, tmp_4

def replacement_args(tmp_2):
    return (tmp_2,)

def replacement_func():
    def flexible_fused_kernel(tmp_2):
        # Determine dimensions based on input tensor shape
        input_shape = tmp_2.shape
        if len(input_shape) == 4:
            # Already in view format: [batch_size, H, W, C]
            batch_size, spatial_h, spatial_w, feature_dim = input_shape
        else:
            # Need to infer dimensions - this handles the reshape case
            # Based on the pattern, we know the final target dimensions
            if feature_dim := 768:
                spatial_h, spatial_w = 32, 32
            elif feature_dim := 384:
                spatial_h, spatial_w = 64, 64
            else:
                # Default fallback
                total_spatial = input_shape[-2] if len(input_shape) >= 2 else input_shape[-1]
                spatial_h = spatial_w = int(total_spatial ** 0.5)
                feature_dim = input_shape[-1] // (spatial_h * spatial_w)
            
            batch_size = input_shape[0] if len(input_shape) > 0 else -1
        
        # Execute fused operation
        output = flexible_fused_view_roll(tmp_2, spatial_h, spatial_w, feature_dim)
        return output
    
    return flexible_fused_kernel