import torch
import triton
import triton.language as tl

def pattern(in_3, in_2):
    # Pattern matches the spatial manipulation sequence for the third variant
    # This will match the third graph pattern (384 channels, 35x35 -> 32x32)
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    output = in_2 + tmp_7
    return output

def replacement_args(in_3, in_2):
    # Extract arguments for the optimized kernel
    return (in_3, in_2)

@triton.jit
def spatial_manipulation_kernel(
    input_ptr,
    residual_ptr,
    output_ptr,
    n_elements,
    H_orig, W_orig, H_new, W_new, C,
    roll_shift_h, roll_shift_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program processes a block of the output
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate the original spatial coordinates
    idx = offsets
    spatial_idx = idx // C
    c_idx = idx % C
    
    # Convert linear spatial index to spatial coordinates
    h_idx = spatial_idx // W_new
    w_idx = spatial_idx % W_new
    
    # Calculate original coordinates before slicing and rolling
    # The transformation is: view(-1, H_orig, W_orig, C) -> roll -> slice -> view(1, H_new*W_new, C)
    h_orig = h_idx + roll_shift_h
    w_orig = w_idx + roll_shift_w
    
    # Handle rolling by wrapping around
    h_orig = h_orig % H_orig
    w_orig = w_orig % W_orig
    
    # Calculate original linear index
    orig_linear_idx = h_orig * W_orig * C + w_orig * C + c_idx
    
    # Load input data directly from original location
    input_val = tl.load(input_ptr + orig_linear_idx, mask=tl.broadcast_to(mask, orig_linear_idx.shape), other=0.0)
    residual_val = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    
    # Add residual
    output_val = input_val + residual_val
    
    # Store result
    tl.store(output_ptr + offsets, output_val, mask=mask)

@torch.fx.wrap
def optimized_spatial_manipulation(in_3, in_2):
    # Use the 384-channel pattern dimensions
    H_orig, W_orig = 35, 35
    H_new, W_new = 32, 32
    C = 384
    total_elements = 1 * 32 * 32 * 384  # 1024
    roll_shift_h, roll_shift_w = 3, 3
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(in_2)
    
    spatial_manipulation_kernel[(num_programs,)](
        input_ptr=in_3,
        residual_ptr=in_2,
        output_ptr=output,
        n_elements=total_elements,
        H_orig=H_orig, W_orig=W_orig, H_new=H_new, W_new=W_new, C=C,
        roll_shift_h=roll_shift_h, roll_shift_w=roll_shift_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_spatial_manipulation