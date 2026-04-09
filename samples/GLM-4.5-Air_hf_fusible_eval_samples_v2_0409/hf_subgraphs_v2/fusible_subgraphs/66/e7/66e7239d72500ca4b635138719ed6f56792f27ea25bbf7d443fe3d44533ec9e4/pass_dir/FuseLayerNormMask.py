import torch
import triton
import triton.language as tl

# Pattern matching function for layer_norm + multiplication
def layer_norm_pattern(input_tensor, norm_weight, norm_bias, attention_mask_float):
    """
    Match the sequence: layer_norm -> multiply_with_expanded_mask
    Args:
        input_tensor: [1, 16, 768], torch.float16/bfloat16
        norm_weight: [768], torch.float16/bfloat16
        norm_bias: [768], torch.float16/bfloat16
        attention_mask_float: [1, 16, 768], torch.float32
    Returns:
        layer_norm_output: [1, 16, 768], torch.float16/bfloat16
        masked_output: [1, 16, 768], torch.float16/bfloat16
    """
    tmp_4 = torch.nn.functional.layer_norm(input_tensor, (768,), norm_weight, norm_bias, 1e-12)
    tmp_8 = tmp_4 * attention_mask_float
    return tmp_4, tmp_8

def replacement_args(input_tensor, norm_weight, norm_bias, attention_mask_float):
    return (input_tensor, norm_weight, norm_bias, attention_mask_float)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, gamma_ptr, 
    batch, seq_len, hidden_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Simplified Triton layer normalization kernel
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_offset < batch
    mask_n = n_offset < hidden_size
    
    # Load x, weight, bias
    x_batch_seq = x_ptr + m_offset[:, None] * seq_len * hidden_size + n_offset[None, :]
    x = tl.load(x_batch_seq, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    weight = tl.load(weight_ptr + n_offset, mask=mask_n, other=1.0)
    bias = tl.load(bias_ptr + n_offset, mask=mask_n, other=0.0)
    
    # Layer norm computation (simplified)
    eps = 1e-12
    mean = tl.sum(x, axis=1) / seq_len
    var = tl.sum((x - mean[:, None]) ** 2, axis=1) / seq_len
    x_norm = (x - mean[:, None]) / tl.sqrt(var[:, None] + eps)
    out = x_norm * weight[None, :] + bias[None, :]
    
    return out, m_offset, pid_n * BLOCK_N, mask_m

@triton.jit
def mask_kernel(
    layer_norm_output_ptr,
    attention_mask_ptr,
    output_ptr,
    batch, seq_len, hidden_size,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Apply attention mask to layer norm output
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_offset < batch
    mask_n = n_offset < hidden_size
    
    # Load layer norm output
    ln_batch_seq = layer_norm_output_ptr + m_offset[:, None] * seq_len * hidden_size + n_offset[None, :]
    layer_norm_out = tl.load(ln_batch_seq, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Load attention mask (convert to input dtype)
    mask_batch_seq = attention_mask_ptr + m_offset[:, None] * seq_len * hidden_size + n_offset[None, :]
    attention_mask = tl.load(mask_batch_seq, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Apply mask
    output = layer_norm_out * attention_mask
    
    return output, m_offset, pid_n * BLOCK_N, mask_m

@torch.fx.wrap
def fused_layer_norm_mask(input_tensor, norm_weight, norm_bias, attention_mask_float):
    """
    Combined layer normalization and masking operation
    Args:
        input_tensor: [1, 16, 768], torch.float16/bfloat16
        norm_weight: [768], torch.float16/bfloat16
        norm_bias: [768], torch.float16/bfloat16
        attention_mask_float: [1, 16, 768], torch.float32
    Returns:
        layer_norm_output: [1, 16, 768], torch.float16/bfloat16
        masked_output: [1, 16, 768], torch.float16/bfloat16
    """
    batch, seq_len, hidden_size = input_tensor.shape
    
    # Optimize block sizes based on tensor dimensions
    BLOCK_M = 1  # Process one batch at a time
    BLOCK_N = 256  # Hidden dimension block size
    
    num_batch_blocks = (batch + BLOCK_M - 1) // BLOCK_M
    num_hidden_blocks = (hidden_size + BLOCK_N - 1) // BLOCK_N
    
    # Create output tensors
    layer_norm_output = torch.empty_like(input_tensor)
    masked_output = torch.empty_like(input_tensor)
    
    # Launch kernel for layer norm
    layer_norm_kernel[(num_batch_blocks, num_hidden_blocks)](
        x_ptr=input_tensor,
        weight_ptr=norm_weight,
        bias_ptr=norm_bias,
        gamma_ptr=torch.ones_like(norm_weight),  # placeholder for gamma
        batch=batch,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    # Launch kernel for masking 
    mask_kernel[(num_batch_blocks, num_hidden_blocks)](
        layer_norm_output_ptr=layer_norm_output,
        attention_mask_ptr=attention_mask_float,
        output_ptr=masked_output,
        batch=batch,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return layer_norm_output, masked_output

def replacement_func():
    return fused_layer_norm_mask