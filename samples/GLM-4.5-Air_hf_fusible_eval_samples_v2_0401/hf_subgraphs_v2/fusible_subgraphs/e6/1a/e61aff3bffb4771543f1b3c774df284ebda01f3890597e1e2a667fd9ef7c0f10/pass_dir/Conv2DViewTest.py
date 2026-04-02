import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern: Conv2D → View (no softmax)"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(conv2d.shape[0], 1, -1)  # Reshape to [batch, 1, seq_len]
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def conv2d_view_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr, 
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Conv2D + View kernel"""
    batch_idx = tl.program_id(0)
    seq_len = height * width
    
    # Calculate Conv2D output for each position: weighted sum across channels + bias
    for pos in range(0, seq_len, BLOCK_SIZE):
        pos_offsets = pos + tl.arange(0, BLOCK_SIZE)
        pos_mask = pos_offsets < seq_len
        
        # Initialize conv output for this block
        conv_outputs = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Compute weighted sum across channels for each position
        for c in range(in_channels):
            # Load weight for this channel
            weight_c = tl.load(weight_ptr + c)
            
            # Process each position in the block
            for i, local_pos in enumerate(pos_offsets):
                if pos_mask[i]:
                    input_offset = (batch_idx * in_channels * height * width + 
                                  c * height * width + local_pos)
                    input_val = tl.load(input_ptr + input_offset)
                    conv_outputs[i] += weight_c * input_val
        
        # Add bias to all outputs in block
        bias_val = tl.load(bias_ptr)
        conv_outputs += bias_val
        
        # Store results for [batch, 1, seq_len] shape
        for i, local_pos in enumerate(pos_offsets):
            if pos_mask[i]:
                # Output shape: [batch, 1, seq_len] 
                output_offset = (batch_idx * 1 * seq_len + 0 * seq_len + local_pos)
                tl.store(output_ptr + output_offset, conv_outputs[i])

@torch.fx.wrap
def conv2d_view_ops(bias, weight, input_tensor):
    """Conv2D + View wrapper"""
    batch_size, in_channels, height, width = input_tensor.shape
    
    # Output shape: [batch, 1, height*width]
    out_seq_len = height * width
    output = torch.empty((batch_size, 1, out_seq_len), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    BLOCK_SIZE = 256
    grid_size = (batch_size,)
    
    conv2d_view_kernel[grid_size](
        bias,
        weight, 
        input_tensor,
        output,
        batch_size,
        in_channels,
        height,
        width, 
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the fused kernel function"""
    return conv2d_view_ops