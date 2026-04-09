import torch
import triton
import triton.language as tl

def pattern(in_3):
    # Match the exact operations from the model
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    return tmp_5, tmp_7

def replacement_args(in_3, roll_shifts=(3, 3), crop_slice=(slice(None, 32, None), slice(None, 32, None))):
    # Extract original input shape to compute target dimensions
    original_shape = in_3.shape
    target_H = 35
    crop_H = 32
    target_C = 384
    target_seq_len = 1024
    
    return (in_3, original_shape, (roll_shifts, crop_slice), (target_H, crop_H, target_C, target_seq_len))

@triton.jit
def spatial_transform_kernel(
    input_ptr,
    output_ptr,
    n_total_elements,
    spatial_dims,
    roll_shifts,
    target_dims,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n_programs = tl.cdiv(n_total_elements, BLOCK_SIZE)
    
    # Each program handles BLOCK_SIZE consecutive elements
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_total_elements
    
    # Calculate spatial indices based on flat index
    H, W, C = spatial_dims
    
    # Compute linearized indices into the target tensor
    idx = offsets
    target_H, target_W, target_C = target_dims[:-1]
    
    # Reconstruct spatial coordinates
    b = idx // (target_H * target_W * target_C)
    h = (idx // (target_W * target_C)) % target_H
    w = (idx // target_C) % target_W
    c = idx % target_C
    
    # Add roll offsets manually
    roll_h, roll_w = roll_shifts
    h = (h + roll_h) % target_H
    w = (w + roll_w) % target_W
    
    # Linearized index into input tensor
    input_idx = b * (H * W * C) + h * (W * C) + w * C + c
    input_mask = input_idx < n_total_elements
    
    # Load input data
    input_data = tl.load(input_ptr + input_idx, mask=input_mask, other=0.0)
    
    # Store output data directly
    tl.store(output_ptr + offsets, input_data, mask=mask)

@torch.fx.wrap
def optimized_spatial_transform(input_tensor, original_shape, roll_crop_info, target_dims):
    # Roll shifts and crop slice
    roll_shifts, crop_slice = roll_crop_info
    target_H, crop_H, target_C, target_seq_len = target_dims
    
    # Calculate output shape after crop
    output_shape = (1, crop_H, crop_H, target_C)
    total_elements = crop_H * crop_H * target_C
    
    # Create output tensor
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid_size = (tl.cdiv(total_elements, BLOCK_SIZE),)
    
    spatial_dims = (target_H, target_H, target_C)
    target_spatial_dims = (crop_H, crop_H, target_C)
    
    spatial_transform_kernel[grid_size](
        input_tensor,
        output_tensor,
        total_elements,
        spatial_dims,
        roll_shifts,
        target_spatial_dims,
        BLOCK_SIZE
    )
    
    return output_tensor.contiguous().view(1, target_seq_len, target_C)

def replacement_func():
    return optimized_spatial_transform