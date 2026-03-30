import torch
import triton
import triton.language as tl

def pattern(tensor, add_tensor):
    # Match the sequence: contiguous -> view -> roll -> slice -> contiguous -> view + addition
    tmp_2 = tensor.contiguous()
    tmp_3 = tmp_2.view(1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    tmp_8 = add_tensor + tmp_7
    return tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8

def replacement_args(tensor, add_tensor):
    return (tensor, add_tensor)

@triton.jit
def fused_manipulation_add_kernel(
    input_ptr, 
    add_ptr, 
    output_ptr,
    hidden_size: tl.constexpr,
    batch_size: tl.constexpr,
    orig_height: tl.constexpr,
    orig_width: tl.constexpr,
    slice_height: tl.constexpr,
    slice_width: tl.constexpr,
    roll_shift_h: tl.constexpr,
    roll_shift_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    total_elements = batch_size * slice_height * slice_width * hidden_size
    elements_per_program = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_idx = pid * elements_per_program
    end_idx = min((pid + 1) * elements_per_program, total_elements)
    
    for idx in range(start_idx, end_idx):
        batch = idx // (slice_height * slice_width * hidden_size)
        slice_idx = 0  # Only one batch and slice in this case
        
        local_idx = idx % (slice_height * slice_width * hidden_size)
        h = (local_idx // (slice_width * hidden_size)) % slice_height
        w = (local_idx // hidden_size) % slice_width
        c = local_idx % hidden_size
        
        # Calculate original position before slice and roll
        orig_h = h + roll_shift_h
        orig_h = orig_h % orig_height
        
        orig_w = w + roll_shift_w
        orig_w = orig_w % orig_width
        
        # Calculate input offset - flatten original 5D tensor to 4D then access
        # Original tensor shape: [1, 5, 7, 5, 7, 384] -> view to [1, 35, 35, 384]
        input_offset = (batch * orig_height + orig_h) * orig_width * hidden_size + orig_w * hidden_size + c
        
        # Load from input tensor
        tensor_value = tl.load(input_ptr + input_offset, other=0.0)
        
        # Load from addition tensor - this should be [1, 1024, 384]
        # Add offset: batch * seq_len * hidden_size + seq_idx * hidden_size + hidden_idx
        add_offset = batch * slice_height * slice_width * hidden_size + (h * slice_width + w) * hidden_size + c
        add_value = tl.load(add_ptr + add_offset, other=0.0)
        
        # Add and store
        result = tensor_value + add_value
        tl.store(output_ptr + idx, result, allow_overlap=True)

@torch.fx.wrap
def fused_manipulation_add_op(tensor, add_tensor):
    # Shape information for this specific pattern
    batch_size = 1
    hidden_size = 384
    orig_height, orig_width = 35, 35  # After view operation
    slice_height, slice_width = 32, 32  # After slicing
    
    # Check if match (this helps identify when to apply the fusion)
    if tensor.shape != (1, 5, 7, 5, 7, 384) or add_tensor.shape != (1, 1024, 384):
        # Fall back to original computation if shapes don't match
        tmp_2 = tensor.contiguous()
        tmp_3 = tmp_2.view(1, 35, 35, 384)
        tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
        tmp_5 = tmp_4[(slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None))]
        tmp_6 = tmp_5.contiguous()
        tmp_7 = tmp_6.view(1, 1024, 384)
        tmp_8 = add_tensor + tmp_7
        return tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, tmp_7, tmp_8
    
    # Create output tensor
    output_size = batch_size * slice_height * slice_width * hidden_size
    output = torch.empty(batch_size, slice_height, slice_width, hidden_size, device=tensor.device, dtype=tensor.dtype)
    
    # Launch Triton kernel
    n_elements = output_size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_manipulation_add_kernel[(num_programs,)](
        input_ptr=tensor,
        add_ptr=add_tensor,
        output_ptr=output.data_ptr(),  # Use data_ptr() for raw pointer access
        hidden_size=hidden_size,
        batch_size=batch_size,
        orig_height=orig_height,
        orig_width=orig_width,
        slice_height=slice_height,
        slice_width=slice_width,
        roll_shift_h=3,
        roll_shift_w=3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to match expected output pattern
    tmp_7 = output.view(1, 1024, 384)
    tmp_8 = add_tensor + tmp_7  # This addition won't be optimized in this version
    
    return tensor, tensor.view(1, 35, 35, 384), None, output, output, tmp_7, tmp_8

def replacement_func():
    return fused_manipulation_add_op