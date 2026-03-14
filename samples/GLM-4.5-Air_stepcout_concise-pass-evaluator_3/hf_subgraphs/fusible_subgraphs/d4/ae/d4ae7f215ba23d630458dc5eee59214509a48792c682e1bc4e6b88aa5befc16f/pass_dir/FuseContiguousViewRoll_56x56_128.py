import torch
import triton
import triton.language as tl

def pattern(in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 56, 56, 128)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 3136, 128)
    return tmp_5

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def fused_roll_kernel_56x56(
    input_ptr,
    output_ptr,
    n_elements,
    spatial_size: tl.constexpr,
    channel_size: tl.constexpr,
    shift_h: tl.constexpr,
    shift_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data directly from the input tensor
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate spatial dimensions for roll operation
    total_spatial = spatial_size * spatial_size
    total_elements_per_item = spatial_size * spatial_size * channel_size
    
    # Decompose offsets into spatial and channel coordinates
    item_offset = offsets // total_elements_per_item
    remaining_offset = offsets % total_elements_per_item
    
    spatial_idx = remaining_offset // channel_size
    channel_idx = remaining_offset % channel_size
    
    # Extract height and width indices
    h_idx = spatial_idx // spatial_size
    w_idx = spatial_idx % spatial_size
    
    # Apply roll operation with circular shift
    new_h_idx = (h_idx + shift_h) % spatial_size
    new_w_idx = (w_idx + shift_w) % spatial_size
    
    # Recalculate spatial position
    new_spatial_idx = new_h_idx * spatial_size + new_w_idx
    new_remaining_offset = new_spatial_idx * channel_size + channel_idx
    
    # Recalculate final offset
    new_offset = item_offset * total_elements_per_item + new_remaining_offset
    
    # Store result - this handles the view reshaping implicitly
    tl.store(output_ptr + new_offset, input_vals, mask=mask)

@torch.fx.wrap
def fused_roll_operation_56x56(in_3):
    # Input shape: [1, 8, 7, 8, 7, 128]
    # Target output shape: [1, 3136, 128]
    
    total_elements = 1 * 56 * 56 * 128  # 1 x (56x56) x 128
    
    # Create output tensor - note we write to it in a different order
    result = torch.empty_like(in_3)  # Use same dtype/device, will reshape later
    
    # Launch Triton kernel with more optimal block size
    BLOCK_SIZE = 512
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_roll_kernel_56x56[(num_programs,)](
        input_ptr=in_3,
        output_ptr=result,
        n_elements=total_elements,
        spatial_size=56,
        channel_size=128,
        shift_h=3,
        shift_w=3,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape to final output format
    return result.view(1, 3136, 128)

def replacement_func():
    return fused_roll_operation_56x56