import torch
import triton
import triton.language as tl

def pattern(in_3, H, W, C, shift_h, shift_w):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, H, W, C)
    tmp_4 = torch.roll(tmp_3, shifts=(shift_h, shift_w), dims=(1, 2))
    tmp_5 = tmp_4.view(1, H * W, C)
    return tmp_5

def replacement_args(in_3, H, W, C, shift_h, shift_w, HW):
    return (in_3, H, W, C, shift_h, shift_w, HW)

@triton.jit
def fused_view_roll_view_kernel(
    input_ptr,
    output_ptr,
    orig_H, orig_W, orig_C,
    H, W, C,
    shift_h, shift_w,
    batch_size, HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    input_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate the required input size
    total_HW = orig_H * orig_W
    input_size = batch_size * total_HW * orig_C
    
    mask = input_offset < input_size
    orig_idx = input_offset
    
    # Convert linear index to original (b, h, w, c) coordinates
    batch_idx = orig_idx // (total_HW * orig_C)
    rem_idx = orig_idx % (total_HW * orig_C)
    h_orig = rem_idx // (orig_W * orig_C)
    rem_idx2 = rem_idx % (orig_W * orig_C)
    w_orig = rem_idx2 // orig_C
    c_orig = rem_idx2 % orig_C
    
    # Apply roll operation with proper bounds checking
    h_new = (h_orig + shift_h) % H
    w_new = (w_orig + shift_w) % W
    
    # Calculate new linear index
    new_idx = batch_idx * (H * W * C) + h_new * (W * C) + w_new * C + c_orig
    
    # Load output and input tensors
    input_val = tl.load(input_ptr + orig_idx, mask=mask, other=0.0)
    output_val = tl.load(output_ptr + new_idx, mask=mask, other=0.0)
    
    # Store the rolled result
    tl.store(output_ptr + new_idx, input_val, mask=mask)

@torch.fx.wrap
def fused_view_roll_ops(in_3, H, W, C, shift_h, shift_w, HW):
    orig_shape = in_3.shape
    batch_size = orig_shape[0]
    orig_H = orig_shape[1] if len(orig_shape) == 6 else orig_shape[1]
    orig_W = orig_shape[2] if len(orig_shape) == 6 else orig_shape[3] 
    orig_C = orig_shape[5] if len(orig_shape) == 6 else orig_shape[2]
    
    input_size = in_3.numel()
    BLOCK_SIZE = 1024
    num_programs = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(in_3)
    
    fused_view_roll_view_kernel[(num_programs,)](
        input_ptr=in_3,
        output_ptr=output,
        orig_H=orig_H, orig_W=orig_W, orig_C=orig_C,
        H=H, W=W, C=C,
        shift_h=shift_h, shift_w=shift_w,
        batch_size=batch_size, HW=HW,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.view(1, H * W, C)

def replacement_func():
    return fused_view_roll_ops