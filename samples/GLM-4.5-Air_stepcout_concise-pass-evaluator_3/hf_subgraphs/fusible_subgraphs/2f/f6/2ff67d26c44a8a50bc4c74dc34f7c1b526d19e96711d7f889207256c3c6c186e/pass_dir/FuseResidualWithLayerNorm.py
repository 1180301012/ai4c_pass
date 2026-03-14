import torch
import triton
import triton.language as tl

def pattern(in_2, in_3, in_0, in_1):
    """Pattern matching for residual connection + LayerNorm + scale + bias"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3 + tmp_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5  # Recompute since tmp_4 and tmp_5 are used later
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = tmp_1 * tmp_13
    tmp_15 = tmp_14 + tmp_0
    return tmp_15

def replacement_args(in_2, in_3, in_0, in_1):
    return (in_2, in_3, in_0, in_1)

@triton.jit
def layernorm_kernel(
    x_ptr,  # input tensor [batch, seq, hidden]
    residual_ptr,  # residual tensor [batch, seq, hidden] 
    weight_ptr,  # weight tensor [hidden]
    bias_ptr,    # bias tensor [hidden]
    out_ptr,     # output tensor [batch, seq, hidden]
    batch,
    seq_len,
    hidden_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
):
    # Program IDs for 2D grid - each program handles one element in batch x seq
    pid_m = tl.program_id(0)  # batch position
    pid_n = tl.program_id(1)  # seq position
    
    # Check bounds
    if pid_m >= batch or pid_n >= seq_len:
        return
    
    # Calculate offset for current position
    offset = pid_m * seq_len * hidden_size + pid_n * hidden_size
    
    # Create mask for actual hidden size (768)
    mask = tl.arange(0, HIDDEN_SIZE) < hidden_size
    
    # Load weight and bias (only first hidden_size elements)
    weight = tl.load(weight_ptr + tl.arange(0, HIDDEN_SIZE), mask=mask)
    bias = tl.load(bias_ptr + tl.arange(0, HIDDEN_SIZE), mask=mask)
    
    # Load input and residual for current position
    x = tl.load(x_ptr + offset + tl.arange(0, HIDDEN_SIZE), mask=mask)
    residual = tl.load(residual_ptr + offset + tl.arange(0, HIDDEN_SIZE), mask=mask)
    
    # Fuse residual connection
    fused_input = residual + x
    
    # Compute mean over hidden dimension (LayerNorm normalization)
    mean = tl.sum(fused_input) / hidden_size
    
    # Center the values and compute variance over hidden dimension
    centered = fused_input - mean
    variance = tl.sum(centered * centered) / hidden_size
    
    # Compute standard deviation with epsilon
    std = tl.sqrt(variance + 1e-07)
    
    # Normalize
    normalized = centered / std
    
    # Scale and shift
    result = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offset + tl.arange(0, HIDDEN_SIZE), result, mask=mask)

@torch.fx.wrap
def fused_layernorm(residual, hidden, weight, bias):
    batch, seq_len, hidden_size = hidden.shape
    
    # Grid: each program handles one (batch, seq) position
    grid = (batch, seq_len)
    
    # Allocate output
    output = torch.empty_like(hidden)
    
    # Launch kernel
    layernorm_kernel[grid](
        residual,
        hidden,
        weight,
        bias,
        output,
        batch,
        seq_len,
        hidden_size,
        1,  # BLOCK_SIZE_M - not used but required
        1,  # BLOCK_SIZE_N - not used but required
        1024,  # HIDDEN_SIZE - use power of 2 (1024 >= 768)
    )
    
    return output

def replacement_func():
    return fused_layernorm