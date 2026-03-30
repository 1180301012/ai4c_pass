import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the exact computation sequence for ConvBert model
    tmp_1 = x.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5

def replacement_args(x):
    return (x,)

@triton.jit
def convbert_fused_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    CHANNEL_GROUPING_FACTOR: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < batch_size * seq_len * 8 * 9
    
    # Process each element
    for i in range(BLOCK_SIZE):
        if mask[i]:
            output_idx = indices[i]
            
            # Map to coordinates: [batch, seq, channel, elem_in_window]
            batch_idx = output_idx // (seq_len * 8 * 9)
            remainder = output_idx % (seq_len * 8 * 9)
            seq_idx = remainder // (8 * 9)
            channel_idx = remainder // 9
            window_idx = remainder % 9
            
            # Map output channel back to original channels (8 -> 16)
            orig_channel = channel_idx * CHANNEL_GROUPING_FACTOR
            
            # Calculate the input position for this window element
            # The window spans from seq_idx - PADDING to seq_idx - PADDING + WINDOW_SIZE - 1
            # Each output element corresponds to a specific position within this window
            input_seq_pos = seq_idx - PADDING + window_idx
            
            # Load input with boundary checking
            if 0 <= input_seq_pos < seq_len:
                input_offset = batch_idx * 16 * seq_len + orig_channel * seq_len + input_seq_pos
                value = tl.load(x_ptr + input_offset)
            else:
                # Padding with zeros
                value = tl.float16(0.0)
            
            # Store result
            tl.store(out_ptr + output_idx, value)

@torch.fx.wrap
def convbert_fused_operation(x):
    # ConvBert specific dimensions: [1, 16, 45]
    batch_size = x.shape[0]
    channels = 16
    seq_len = x.shape[2]
    
    # Output dimensions: [-1, 8, 9] = [batch_size * seq_len * 8, 9]
    total_output_elements = batch_size * seq_len * 8 * 9
    
    # Create output tensor
    out = torch.empty(total_output_elements, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    convbert_fused_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        CHANNEL_GROUPING_FACTOR=2,  # 16 -> 8
        WINDOW_SIZE=9,
        PADDING=4,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to final format
    return out.reshape(batch_size, seq_len, 8, 9).reshape(-1, 8, 9)

def replacement_func():
    return convbert_fused_operation