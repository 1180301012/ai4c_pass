import torch
import triton
import triton.language as tl
import math

def pattern(tmp_11, normalized_shape, weight, bias):
    # Dropout followed by layer normalization
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, normalized_shape, weight, bias, 1e-12)
    return tmp_12, tmp_13

def replacement_args(tmp_11, normalized_shape, weight, bias):
    return (tmp_11, normalized_shape, weight, bias)

@triton.jit
def fused_dropout_layernorm_kernel(
    input_ptr,
    weight_ptr, bias_ptr,
    dropout_output_ptr, ln_output_ptr,
    batch_size, seq_len, hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one position in the sequence
    program_id = tl.program_id(0)
    batch_id = program_id // seq_len
    seq_id = program_id % seq_len
    
    if batch_id >= batch_size or seq_id >= seq_len:
        return
    
    # Calculate base indices
    input_base = (batch_id * seq_len + seq_id) * hidden_dim
    dropout_base = input_base
    ln_base = input_base
    
    # Load input with tiling
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    # Load input data
    input_data = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0)
    
    # Apply dropout (p=0.1, training=False means no dropout actually applied, keeping for completeness)
    dropout_mask = (tl.rand(offsets) > 0.1) if tl.program_id(1) == 0 else True
    dropout_output = input_data * dropout_mask
    
    # Store dropout output
    tl.store(dropout_output_ptr + dropout_base + offsets, dropout_output, mask=mask)
    
    # Layer normalization calculation
    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate mean and variance with Welford's algorithm for numerical stability
    eps = 1e-12
    local_mean = tl.sum(input_data) / hidden_dim
    local_var = tl.sum((input_data - local_mean) * (input_data - local_mean)) / hidden_dim
    
    # Normalize and apply scale/bias
    normalized = (input_data - local_mean) * tl.rsqrt(local_var + eps)
    ln_output = normalized * weight + bias
    
    # Store layer norm output
    tl.store(ln_output_ptr + ln_base + offsets, ln_output, mask=mask)

@torch.fx.wrap 
def fused_dropout_layernorm(input, normalized_shape, weight, bias):
    batch_size, seq_len, hidden_dim = input.shape
    
    # Set optimal block size
    BLOCK_SIZE = min(1024, hidden_dim)
    
    # Calculate grid size
    total_elements = batch_size * seq_len
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    dropout_output = torch.empty_like(input)
    ln_output = torch.empty_like(input)
    
    # Launch kernel (using a 1D grid for simplicity, dropout could be conditional)
    fused_dropout_layernorm_kernel[(num_programs, 1)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        dropout_output_ptr=dropout_output,
        ln_output_ptr=ln_output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dropout_output, ln_output

def replacement_func():
    return fused_dropout_layernorm