import torch
import triton
import triton.language as tl

def pattern(inv_freq, seq_input):
    tmp_0 = inv_freq
    tmp_1 = torch.arange(64, device=torch.device('cuda:0'))
    tmp_2 = tmp_1.type_as(tmp_0)
    tmp_3 = torch.outer(tmp_2, tmp_0)
    tmp_4 = torch.cat((tmp_3, tmp_3), dim=-1)
    tmp_5 = tmp_4.to(torch.device('cuda:0'))
    tmp_6 = tmp_5.cos()
    tmp_7 = tmp_6[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_8 = tmp_5.sin()
    tmp_9 = tmp_8[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_10 = tmp_7[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, None, None)]
    tmp_11 = tmp_9[slice(None, None, None), slice(None, None, None), slice(None, 64, None), slice(None, None, None)]
    tmp_12 = seq_input * tmp_10
    tmp_13 = seq_input.chunk(2, dim=-1)
    tmp_14 = tmp_13[0]
    tmp_15 = tmp_13[1]
    return tmp_7, tmp_9, tmp_11, tmp_12, tmp_14, tmp_15

def replacement_args(inv_freq, seq_input):
    return (inv_freq, seq_input)

@triton.jit
def rotary_embedding_kernel(
    inv_freq_ptr,
    cos_output_ptr,
    sin_output_ptr,
    sequence_len: tl.constexpr,
):
    # Each program handles one row (position) of the output
    row_idx = tl.program_id(0)
    mask = row_idx < sequence_len
    
    # Load frequency tensor (32 elements)
    inv_freq = tl.load(inv_freq_ptr + tl.arange(0, 32))
    
    # Create position for this row
    pos = tl.broadcast_to(row_idx, (32,))
    
    # Compute outer product element-wise: pos[i] * inv_freq[i]
    # This gives us [32] for the first half
    first_half = pos * inv_freq
    
    # Duplicate for second half (concatenation with itself)
    second_half = first_half
    full_result = tl.concatenate([first_half, second_half], axis=0)
    
    # Compute cos and sin
    cos_vals = tl.cos(full_result)
    sin_vals = tl.sin(full_result)
    
    # Store cos and sin values in flattened arrays
    for i in range(64):
        if mask:  # Only store if this row is valid
            cos_ptr = cos_output_ptr + row_idx * 64 + i
            sin_ptr = sin_output_ptr + row_idx * 64 + i
            tl.store(cos_ptr, cos_vals[i])
            tl.store(sin_ptr, sin_vals[i])

@torch.fx.wrap
def fused_rotary_embedding(inv_freq, seq_input):
    # For now, use hardcoded 64, but this should be extracted from model
    seq_len = 64
    
    # Create output tensors with correct shape [1, 1, seq_len, 64]
    cos_output = torch.zeros((1, 1, seq_len, 64), dtype=torch.float32, device=inv_freq.device)
    sin_output = torch.zeros((1, 1, seq_len, 64), dtype=torch.float32, device=inv_freq.device)
    
    # Flatten outputs for easier kernel access [seq_len * 64]
    cos_flat = cos_output.reshape(-1)
    sin_flat = sin_output.reshape(-1)
    
    # Launch kernel - one program per row (position)
    num_programs = seq_len
    rotary_embedding_kernel[(num_programs,)](
        inv_freq_ptr=inv_freq,
        cos_output_ptr=cos_flat,
        sin_output_ptr=sin_flat,
        sequence_len=seq_len,
    )
    
    # Compute sliced versions
    cos_sliced = cos_output[slice(None, None, None), slice(None, None, None), slice(None, seq_len, None), slice(None, None, None)]
    sin_sliced = sin_output[slice(None, None, None), slice(None, None, None), slice(None, seq_len, None), slice(None, None, None)]
    
    # Compute multiplication with input 
    result_mult = seq_input * cos_sliced
    
    # Compute chunked parts  
    chunked = seq_input.chunk(2, dim=-1)
    chunked_0 = chunked[0]
    chunked_1 = chunked[1]
    
    return cos_output, sin_output, sin_sliced, result_mult, chunked_0, chunked_1

def replacement_func():
    return fused_rotary_embedding