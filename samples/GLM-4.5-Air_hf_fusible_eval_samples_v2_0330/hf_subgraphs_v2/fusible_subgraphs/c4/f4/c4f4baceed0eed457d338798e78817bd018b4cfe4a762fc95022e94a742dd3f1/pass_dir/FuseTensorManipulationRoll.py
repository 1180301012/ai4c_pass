import torch
import triton
import triton.language as tl

def pattern(tensor):
    # Match the sequence: contiguous -> view(1, 35, 35, 384) -> roll(shifts=(3,3), dims=(1,2)) -> slice -> contiguous -> view
    tmp_2 = tensor.contiguous()
    tmp_3 = tmp_2.view(1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    return tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7

def replacement_args(tensor):
    return (tensor,)

@triton.jit
def fused_manipulation_kernel(
    input_ptr, output_ptr,
    n_slices: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    channels: tl.constexpr,
    slice_height: tl.constexpr,
    slice_width: tl.constexpr,
    roll_shift_h: tl.constexpr,
    roll_shift_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    total_elements = slice_height * slice_width * channels
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    for idx in range(start_idx, end_idx):
        slice_idx = idx // (slice_width * channels)
        h = (idx // channels) % slice_height
        w = idx % channels
        
        # Calculate original position before slice and roll
        orig_h = h + roll_shift_h
        orig_h = orig_h % height
        
        orig_w = w + roll_shift_w
        orig_w = orig_w % width
        
        # Calculate input offset
        input_offset = (slice_idx * height + orig_h) * width * channels + w
        output_offset = idx
        
        # Load from input (simulated contiguous access)
        value = tl.load(input_ptr + input_offset, other=0.0)
        # Store to output
        tl.store(output_ptr + output_offset, value, allow_overlap=True)

@torch.fx.wrap
def fused_manipulation_op(tensor):
    # Get input shape info
    orig_shape = tensor.shape
    # For this pattern: [1, 35, 35, 384] -> roll -> slice [1, 32, 32, 384] -> view [1, 1024, 384]
    height, width, channels = 35, 35, 384
    slice_height, slice_width = 32, 32
    
    # Calculate output size
    output_size = slice_height * slice_width * channels
    
    # Create output tensor
    output = torch.empty(1, slice_height, slice_width, channels, device=tensor.device, dtype=tensor.dtype)
    
    # Launch Triton kernel
    n_elements = output_size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_manipulation_kernel[(num_programs,)](
        input_ptr=tensor,
        output_ptr=output,
        n_slices=1,
        height=height,
        width=width,
        channels=channels,
        slice_height=slice_height,
        slice_width=slice_width,
        roll_shift_h=3,
        roll_shift_w=3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to final output
    final_output = output.view(1, 1024, 384)
    return output, final_output

def replacement_func():
    return fused_manipulation_op