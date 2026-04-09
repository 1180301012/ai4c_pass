import torch
import triton
import triton.language as tl

# Pattern matching function - this matches the entire computation from input addition to final output
def pattern(in_2, in_3, in_1, in_0):
    """
    Match the fused computation:
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5  # redundant - this will be optimized
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    """
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15

# Argument extraction function
def replacement_args(in_2, in_3, in_1, in_0):
    return (in_2, in_3, in_1, in_0)

# Optimized Triton kernel for fused LayerNorm + Linear
@triton.jit
def fused_layernorm_linear_kernel(
    # Input tensors
    x_ptr,           # in_3 + in_2 
    weight_ptr,      # in_1 (weight)
    bias_ptr,        # in_0 (bias)
    # Output tensor  
    out_ptr,
    # Metadata
    batch_size,
    seq_len,
    hidden_size,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # batch_size dim
    BLOCK_SIZE_N: tl.constexpr,  # hidden_size dim
    # Normalization parameters
    eps: tl.constexpr = 1e-07,
):
    """Fused LayerNorm + Linear computation kernel"""
    # Program identifiers
    pid_m = tl.program_id(0)  # batch index
    pid_n = tl.program_id(1)  # hidden dimension index
    
    # Create ranges for batch and hidden dimensions
    batch_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    hidden_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for valid elements
    batch_mask = batch_offsets < batch_size
    hidden_mask = hidden_offsets < hidden_size
    
    # Load weight and bias vectors for this hidden dimension block
    weight = tl.load(weight_ptr + hidden_offsets, mask=hidden_mask, other=0.0)
    bias = tl.load(bias_ptr + hidden_offsets, mask=hidden_mask, other=0.0)
    
    # Initialize accumulators for mean and variance
    mean_local = tl.zeros(hidden_offsets.shape, dtype=tl.float32)
    var_local = tl.zeros(hidden_offsets.shape, dtype=tl.float32)
    count = tl.zeros(hidden_offsets.shape, dtype=tl.float32)
    
    # First pass: compute mean
    for k in range(0, seq_len):
        # Load input tensor slice: [batch, hidden] for current seq position
        input_ptr = x_ptr + (batch_offsets[:, None] * seq_len * hidden_size + 
                            k * hidden_size + hidden_offsets[None, :])
        
        x = tl.load(input_ptr, mask=batch_mask[:, None] & hidden_mask[None, :], other=0.0)
        
        # Accumulate sum for mean computation
        mean_local += x
        count += batch_mask[:, None]
    
    # Compute mean
    mean = mean_local / seq_len
    
    # Second pass: compute variance
    for k in range(0, seq_len):
        # Load input tensor slice: [batch, hidden] for current seq position
        input_ptr = x_ptr + (batch_offsets[:, None] * seq_len * hidden_size + 
                            k * hidden_size + hidden_offsets[None, :])
        
        x = tl.load(input_ptr, mask=batch_mask[:, None] & hidden_mask[None, :], other=0.0)
        x_centered = x - mean
        x_squared = x_centered * x_centered
        
        # Accumulate variance
        var_local += x_squared
        count += batch_mask[:, None]
    
    # Compute variance and standard deviation
    var = var_local / seq_len
    std = tl.sqrt(var + eps)
    
    # Third pass: normalize and apply linear transformation
    for k in range(0, seq_len):
        # Load input tensor slice: [batch, hidden] for current seq position
        input_ptr = x_ptr + (batch_offsets[:, None] * seq_len * hidden_size + 
                            k * hidden_size + hidden_offsets[None, :])
        
        x = tl.load(input_ptr, mask=batch_mask[:, None] & hidden_mask[None, :], other=0.0)
        
        # Normalize
        x_centered = x - mean
        x_norm = x_centered / std
        
        # Apply linear transformation: output = x_norm * weight + bias
        # Weight and bias are [hidden_size], x_norm is [batch_size, hidden_size]
        # Use proper broadcasting
        output_expanded = x_norm * weight[None, :] + bias[None, :]
        
        # Store result
        output_ptr = out_ptr + (batch_offsets[:, None] * seq_len * hidden_size + 
                               k * hidden_size + hidden_offsets[None, :])
        tl.store(output_ptr, output_expanded, mask=batch_mask[:, None] & hidden_mask[None, :])

@torch.fx.wrap
def fused_layernorm_linear(x, weight, bias):
    """
    High-level wrapper for fused LayerNorm + Linear operation
    x: tensor of shape [batch_size, seq_len, hidden_size] 
    weight: tensor of shape [hidden_size]
    bias: tensor of shape [hidden_size]
    """
    batch_size, seq_len, hidden_size = x.shape
    
    # Determine block sizes based on tensor characteristics
    # For LayerNorm, we want good parallelism across batch and hidden dimensions
    BLOCK_SIZE_M = min(64, batch_size)  # batch dimension block size  
    BLOCK_SIZE_N = min(256, hidden_size)  # hidden dimension block size
    
    # Calculate grid size
    m_blocks = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    n_blocks = (hidden_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor with proper dtype handling
    # Original computation converts to float32 for intermediate operations
    out = torch.empty((batch_size, seq_len, hidden_size), dtype=torch.float32, device=x.device)
    
    # Launch kernel with 2D grid (batch and hidden dimensions)
    fused_layernorm_linear_kernel[(m_blocks, n_blocks)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        eps=1e-07
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    def wrapper(in_2, in_3, in_1, in_0):
        # Compute the input addition first (tmp_3 = in_3 + in_2)
        x = in_3 + in_2
        # Then call the fused LayerNorm + Linear function
        return fused_layernorm_linear(x, in_1, in_0)
    return wrapper