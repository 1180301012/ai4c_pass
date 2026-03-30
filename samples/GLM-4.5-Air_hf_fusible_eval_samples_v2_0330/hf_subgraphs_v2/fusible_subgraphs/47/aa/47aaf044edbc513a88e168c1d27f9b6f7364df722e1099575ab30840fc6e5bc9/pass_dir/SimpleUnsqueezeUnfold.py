import torch
import triton
import triton.language as tl

def pattern(x):
    # Simple pattern that matches just the core operations without conditional logic
    tmp_1 = x.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    
    # Return the transpose result - this optimizes the first 3 operations
    return tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def simple_fused_kernel(
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
            
            # The output shape after unfold + transpose is [batch_size, seq_len, in_channels * 9]
            # So we need to map output_idx to [batch, seq, channel]
            total_features = in_channels * 9
            batch_idx = output_idx // (seq_len * total_features)
            remainder = output_idx % (seq_len * total_features)
            seq_idx = remainder // total_features
            feature_idx = remainder % total_features
            
            # Map feature_idx back to original channel and position within window
            orig_channel = feature_idx // 9
            window_pos = feature_idx % 9
            
            # Calculate the input sequence position
            # unfold with kernel_size=9, padding=4 means it extracts windows
            input_seq_pos = seq_idx - 4 + window_pos
            
            # Load input with boundary checking
            if 0 <= input_seq_pos < seq_len:
                input_offset = batch_idx * in_channels * seq_len + orig_channel * seq_len + input_seq_pos
                value = tl.load(x_ptr + input_offset)
            else:
                # Handle out-of-bounds with padding
                value = tl.float16(0.0)
            
            tl.store(out_ptr + output_idx, value)

@torch.fx.wrap
def simple_fused_operation(x):
    # Get input dimensions
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    seq_len = x.shape[2]
    
    # After unfold + transpose: [batch_size, seq_len, in_channels * 9]
    total_elements = batch_size * seq_len * in_channels * 9
    
    # Create output tensor
    out = torch.empty(total_elements, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_fused_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        seq_len=seq_len,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to final format: [batch_size, seq_len, in_channels * 9]
    return out.reshape(batch_size, seq_len, in_channels * 9)

def replacement_func():
    return simple_fused_operation