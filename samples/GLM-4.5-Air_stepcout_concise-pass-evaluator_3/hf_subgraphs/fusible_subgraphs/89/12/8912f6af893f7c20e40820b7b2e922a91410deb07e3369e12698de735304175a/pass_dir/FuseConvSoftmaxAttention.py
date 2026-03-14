import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern matching: torch.conv2d + view + softmax + unsqueeze"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.view(-1, 1, -1)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv_attention_kernel(
    x_ptr,
    weight_ptr, 
    bias_ptr,
    output_ptr,
    input_batch,
    input_channels,
    input_height,
    input_width,
    output_channels,
    weight_channels,
    seq_len,
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    """Fused Conv2D + attention pattern kernel"""
    # Get program IDs for grid
    batch_channel_pid = tl.program_id(0)
    seq_pid = tl.program_id(1)
    
    batch_idx = batch_channel_pid // output_channels
    out_channel_idx = batch_channel_pid % output_channels
    
    # Calculate sequence range for this program
    seq_start = seq_pid * BLOCK_SIZE_SEQ
    seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE_SEQ)
    mask = seq_offsets < seq_len
    
    # Initialize output with bias
    bias_val = tl.load(bias_ptr + out_channel_idx)
    output_vals = tl.full([BLOCK_SIZE_SEQ], bias_val, dtype=tl.float32)
    
    # Process each spatial position in the sequence
    for seq_offset in tl.arange(0, BLOCK_SIZE_SEQ):
        seq_idx = seq_start + seq_offset
        if seq_idx < seq_len:
            # Convert sequence index back to spatial coordinates
            kh = seq_idx // input_width
            kw = seq_idx % input_width
            
            # Conv2D computation for this spatial position
            result = bias_val
            
            # Process all input channels
            for kc in range(weight_channels):
                # Load weight (weight has shape [output_channels, weight_channels, 1, 1])
                weight_idx = (out_channel_idx * weight_channels + kc)
                weight_val = tl.load(weight_ptr + weight_idx)
                
                # Load input value
                input_idx = ((batch_idx * input_channels + kc) * input_height + kh) * input_width + kw
                input_val = tl.load(x_ptr + input_idx)
                
                # Accumulate result
                result += weight_val * input_val
            
            # Store result
            tl.store(output_vals + seq_offset, result, mask=mask[seq_offset])
    
    # Write output
    output_base = batch_idx * output_channels * seq_len + out_channel_idx * seq_len
    tl.store(output_ptr + output_base + seq_offsets, output_vals, mask=mask)

@torch.fx.wrap  
def fused_conv_attention(in_0, in_1, in_2):
    B, C_in, H_in, W_in = in_2.shape
    C_out, _, _, _ = in_1.shape
    
    # Output spatial dimensions remain the same for stride 1, padding 0 convolution
    H_out = H_in
    W_out = W_in
    
    # Sequence length is flattened spatial dimensions
    seq_len = H_out * W_out
    
    # Create output tensor with shape [B, C_out, seq_len]
    # Note: We'll add the singleton dimension after softmax to match the expected output
    output = torch.empty(B * C_out * seq_len, dtype=in_2.dtype, device=in_2.device)
    
    # Launch grid: (B * C_out) programs for batch-channel pairs, multiple programs for sequence
    grid_size_batch_channel = B * C_out
    
    # Block size for sequence processing - choose to balance occupancy and memory
    BLOCK_SIZE_SEQ = 64
    
    grid = (
        grid_size_batch_channel,
        (seq_len + BLOCK_SIZE_SEQ - 1) // BLOCK_SIZE_SEQ,
    )
    
    fused_conv_attention_kernel[grid](
        in_2.data_ptr(),  # x_ptr
        in_1.data_ptr(),  # weight_ptr
        in_0.data_ptr(),  # bias_ptr
        output.data_ptr(),  # output_ptr
        B,  # input_batch
        C_in,  # input_channels
        H_in,  # input_height
        W_in,  # input_width
        C_out,  # output_channels
        C_in,  # weight_channels
        seq_len,  # seq_len
        BLOCK_SIZE_SEQ,  # BLOCK_SIZE_SEQ
    )
    
    # Reshape output to [B, C_out, seq_len] 
    output_reshaped = output.view(B, C_out, seq_len)
    
    # Add singleton dimension to match original pattern: [B, C_out, 1, seq_len]
    output_singletondim = output_reshaped.unsqueeze(2)
    
    # Apply softmax along the singleton dimension (dim=2) 
    output_softmax = torch.nn.functional.softmax(output_singletondim, dim=2, _stacklevel=5)
    
    # Final unsqueeze at -1 to add trailing singleton dimension
    result = output_softmax.unsqueeze(-1)
    
    return result

def replacement_func():
    return fused_conv_attention