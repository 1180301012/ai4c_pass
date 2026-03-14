import torch
import triton
import triton.language as tl

def layer_norm_pattern(input_tensor, weight_tensor, bias_tensor, eps):
    """Pattern function to match LayerNorm operation"""
    return torch.nn.functional.layer_norm(input_tensor, input_tensor.shape[-1:], weight_tensor, bias_tensor, eps)

def replacement_args(input_tensor, weight_tensor, bias_tensor, eps):
    return (input_tensor, weight_tensor, bias_tensor, eps)

@triton.jit
def optimized_layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the sequence
    batch_idx = tl.program_id(0) // seq_len
    seq_idx = tl.program_id(0) % seq_len
    channel_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    x_offset = (batch_idx * seq_len + seq_idx) * hidden_size + channel_idx
    x = tl.load(x_ptr + x_offset, mask=channel_idx < hidden_size, other=0.0)
    
    # Compute mean and variance
    mean = tl.sum(x) / hidden_size
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / hidden_size
    rstd = 1.0 / tl.sqrt(var + 1e-05)
    
    # Store mean and rstd for potential reuse
    mean_offset = batch_idx * seq_len + seq_idx
    rstd_offset = mean_offset
    tl.store(mean_ptr + mean_offset, mean)
    tl.store(rstd_ptr + rstd_offset, rstd)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < hidden_size, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < hidden_size, other=0.0)
    
    # Apply normalization and scaling
    out = (x_centered * rstd) * weight + bias
    
    # Store output
    tl.store(out_ptr + x_offset, out, mask=channel_idx < hidden_size)

@torch.fx.wrap  
def optimized_layer_norm(input_tensor, weight_tensor, bias_tensor, eps):
    if input_tensor.dim() != 3:
        raise ValueError("Input must be 3D tensor [batch, seq_len, hidden_size]")
    
    batch_size, seq_len, hidden_size = input_tensor.shape
    
    # Create output tensors
    output = torch.empty_like(input_tensor)
    means = torch.empty(batch_size * seq_len, dtype=input_tensor.dtype, device=input_tensor.device)
    rstds = torch.empty(batch_size * seq_len, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Grid setup: one program per element + channel blocks
    grid = (
        batch_size * seq_len,
        (hidden_size + 63) // 64,  # 64 is our BLOCK_SIZE
    )
    
    optimized_layer_norm_kernel[grid](
        input_tensor,
        weight_tensor,
        bias_tensor,
        means,
        rstds,
        output,
        batch_size,
        seq_len,
        hidden_size,
        64,
    )
    
    return output

def replacement_func():
    return optimized_layer_norm