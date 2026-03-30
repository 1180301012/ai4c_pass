import torch
import triton
import triton.language as tl

def pattern(x):
    # Test unsqueeze + unfold pattern
    tmp_1 = x.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def unsqueeze_unfold_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    in_channels,
    seq_len,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < total_elements
    
    for i in range(BLOCK_SIZE):
        if mask[i]:
            output_idx = indices[i]
            
            # unfold output shape: [batch_size, in_channels * 9, seq_len_out]
            # We need to map output_idx to this structure
            features = in_channels * 9
            batch_idx = output_idx // (features * seq_len)
            feature_idx = (output_idx % (features * seq_len)) // seq_len
            seq_idx = (output_idx % (features * seq_len)) % seq_len
            
            # Map feature_idx back to original channel and position in window
            orig_channel = feature_idx // 9
            window_pos = feature_idx % 9
            
            # Calculate input sequence position
            input_seq_pos = seq_idx - 4 + window_pos
            
            # Load input with boundary checking
            if 0 <= input_seq_pos < seq_len:
                input_offset = batch_idx * in_channels * seq_len + orig_channel * seq_len + input_seq_pos
                value = tl.load(x_ptr + input_offset)
            else:
                value = tl.float16(0.0)
            
            tl.store(out_ptr + output_idx, value)

@torch.fx.wrap
def unsqueeze_unfold_operation(x):
    # Get input dimensions
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    seq_len = x.shape[2]
    
    # Calculate output dimensions for unfold
    # unfold output: [batch_size, in_channels * 9, seq_len]
    # We flatten to 1D for processing: [batch_size * in_channels * 9 * seq_len]
    total_elements = batch_size * in_channels * 9 * seq_len
    
    # Create output tensor
    out = torch.empty(total_elements, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    unsqueeze_unfold_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        seq_len=seq_len,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to original unfold output format
    return out.reshape(batch_size, in_channels * 9, seq_len)

def replacement_func():
    return unsqueeze_unfold_operation