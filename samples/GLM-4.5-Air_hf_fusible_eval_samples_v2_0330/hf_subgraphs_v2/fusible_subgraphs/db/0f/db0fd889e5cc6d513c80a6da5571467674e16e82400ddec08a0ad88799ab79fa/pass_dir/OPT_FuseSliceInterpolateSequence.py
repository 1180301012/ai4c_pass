import torch
import triton
import triton.language as tl

def pattern(input_tensor, slice_start_idx, slice_size):
    # Simplified pattern - just the transpose + view + interpolate sequence
    # tmp_17 = tmp_16.transpose(1, 2)
    tmp_17 = input_tensor.transpose(1, 2)
    
    # tmp_18 = tmp_17.view(1, 32, 15, 15)
    tmp_18 = tmp_17.view(1, 32, 15, 15)
    
    # tmp_19 = torch.nn.functional.interpolate(tmp_18, size = (15, 15), mode = 'bicubic', align_corners = False)
    tmp_19 = torch.nn.functional.interpolate(tmp_18, size = (15, 15), mode = 'bicubic', align_corners = False)
    
    # tmp_20 = tmp_19.flatten(2)
    tmp_20 = tmp_19.flatten(2)
    
    # tmp_21 = tmp_20.transpose(1, 2)
    tmp_21 = tmp_20.transpose(1, 2)
    
    return tmp_21

def replacement_args(input_tensor, slice_start_idx, slice_size):
    return (input_tensor, slice_start_idx, slice_size)

@triton.jit
def optimized_interpolate_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_size,
    target_h,
    target_w,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * target_h * target_w * hidden_size:
        return
    
    # Calculate indices
    batch_idx = pid // (target_h * target_w * hidden_size)
    remainder = pid % (target_h * target_w * hidden_size)
    h_idx = remainder // (target_w * hidden_size)
    w_idx = (remainder % (target_w * hidden_size)) // hidden_size
    c_idx = remainder % hidden_size
    
    # Calculate corresponding input position
    input_h_idx = h_idx
    input_w_idx = w_idx
    
    # Load input value
    input_offset = batch_idx * seq_len * hidden_size + input_h_idx * seq_len + input_w_idx * hidden_size + c_idx
    input_val = tl.load(input_ptr + input_offset, mask=c_idx < hidden_size, other=0.0)
    
    # Store output value (no interpolation needed since sizes are same)
    output_offset = batch_idx * target_h * target_w * hidden_size + h_idx * target_w * hidden_size + w_idx * hidden_size + c_idx
    tl.store(output_ptr + output_offset, input_val, mask=c_idx < hidden_size, other=0.0)

@torch.fx.wrap
def optimized_interpolate_sequence(input_tensor):
    # Input shape: [1, 225, 32] (simplified from original)
    batch_size, seq_len, hidden_size = input_tensor.shape
    
    # Simplified: assume input is already in right format [1, 225, 32]
    # Since interpolate with same size is just an identity operation
    
    output_shape = (batch_size, seq_len, hidden_size)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel - simple copy since interpolate size doesn't change
    BLOCK_SIZE_H = 32  # Process all channels at once
    total_elements = batch_size * seq_len * hidden_size
    
    optimized_interpolate_kernel[(total_elements + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,](
        input_tensor,
        output,
        batch_size,
        seq_len,
        hidden_size,
        15,  # target_h (15*15=225)
        15,  # target_w
        BLOCK_SIZE_H,
        64,
    )
    
    return output

def replacement_func():
    return optimized_interpolate_sequence