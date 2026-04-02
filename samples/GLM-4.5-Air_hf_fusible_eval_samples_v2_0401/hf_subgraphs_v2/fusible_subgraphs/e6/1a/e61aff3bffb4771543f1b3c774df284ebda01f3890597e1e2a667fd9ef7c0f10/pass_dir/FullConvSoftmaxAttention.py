import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern: Conv2D(1x1) → View → Softmax(dim=2) → Unsqueeze(-1)"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(conv2d.shape[0], 1, -1)  # Reshape to [batch, 1, seq_len]
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel = 5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return (tmp_5,)  # Return as tuple to match actual graph

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_attention_kernel(
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
    """Fused: 1x1 Conv2D (groups=1) → Softmax → Unsqueeze(-1)"""
    # Each program handles one batch
    batch_idx = tl.program_id(0)
    
    # Calculate sequence length
    seq_len = height * width
    
    # Load bias (single value)
    bias_val = tl.load(bias_ptr)
    
    # Calculate Conv2D output for each position: weighted sum across channels + bias
    # First, compute all Conv2D outputs
    conv_outputs = tl.zeros([seq_len], dtype=tl.float32)
    
    for pos in range(seq_len):
        result = bias_val
        for c in range(in_channels):
            weight_c = tl.load(weight_ptr + c)
            input_offset = (batch_idx * in_channels * height * width + 
                          c * height * width + pos)
            input_val = tl.load(input_ptr + input_offset)
            result += weight_c * input_val
        conv_outputs[pos] = result
    
    # Reshape to [batch, 1, seq_len] for softmax
    # Since we process one batch at a time, we have [seq_len] which represents [1, seq_len] for this batch
    
    # Softmax over the sequence dimension (dim=2 in original, which is the last dim here)
    # Find max for numerical stability
    current_max = tl.max(conv_outputs)
    
    # Compute softmax
    exp_outputs = tl.exp(conv_outputs - current_max)
    exp_sum = tl.sum(exp_outputs)
    softmax_outputs = exp_outputs / exp_sum
    
    # Store results with unsqueeze(-1) - becomes [batch, 1, seq_len, 1]
    for pos in range(seq_len):
        # Output offset: [batch_idx, 0, pos, 0]
        output_offset = (batch_idx * 1 * seq_len * 1 + 0 * seq_len * 1 + pos * 1 + 0)
        tl.store(output_ptr + output_offset, softmax_outputs[pos])

@torch.fx.wrap  
def fused_attention_ops(bias, weight, input_tensor):
    """Fused Conv2D(1x1) + Softmax + Unsqueeze(-1) for attention mechanism"""
    batch_size, in_channels, height, width = input_tensor.shape
    
    # Output shape: [batch, 1, height*width, 1]
    out_seq_len = height * width
    output = torch.empty((batch_size, 1, out_seq_len, 1), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # Launch kernel with optimal block size
    BLOCK_SIZE = 256  # Good for sequence processing
    grid_size = (batch_size,)
    
    fused_attention_kernel[grid_size](
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
    return fused_attention_ops