import torch
import triton
import triton.language as tl

def pattern(dropout_input, ln_weight, ln_bias):
    """Pattern to match dropout followed by layer normalization"""
    tmp_12 = torch.nn.functional.dropout(dropout_input, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (ln_weight.shape[0],), ln_weight, ln_bias, 1e-12)
    return tmp_12, tmp_13

def replacement_args(dropout_input, ln_weight, ln_bias):
    return (dropout_input, ln_weight, ln_bias)

@triton.jit
def fused_dropout_layernorm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    dropout_output_ptr,
    hidden_size,
    batch_size,
    seq_len,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate global position  
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    hidden_idx = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Ensure we don't go out of bounds
    mask = hidden_idx < hidden_size
    
    # Input base pointer for current batch and sequence position
    input_base_ptr = input_ptr + (row_idx * seq_len + col_idx) * hidden_size
    output_base_ptr = output_ptr + (row_idx * seq_len + col_idx) * hidden_size
    dropout_output_base_ptr = dropout_output_ptr + (row_idx * seq_len + col_idx) * hidden_size
    
    # Load input data
    x = tl.load(input_base_ptr + hidden_idx, mask=mask, other=0.0)
    
    # Apply dropout: randomly zero out elements with probability dropout_p
    # Note: For deterministic behavior in inference, dropout should be identity
    # In training mode, this would use random sampling
    dropout_mask = tl.where(tl.rand(hidden_idx.shape) > dropout_p, 1.0, 0.0)
    dropout_x = x * dropout_mask
    
    # Store dropout output
    tl.store(dropout_output_base_ptr + hidden_idx, dropout_x, mask=mask)
    
    # Load layer norm parameters
    weight = tl.load(weight_ptr + hidden_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + hidden_idx, mask=mask, other=0.0)
    
    # Layer norm computation
    # Calculate mean
    mean = tl.sum(dropout_x, axis=0) / hidden_size
    
    # Calculate variance
    x_centered = dropout_x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    
    # Normalize
    epsilon = 1e-12
    std = tl.sqrt(variance + epsilon)
    normalized_x = x_centered / std
    
    # Apply scale and shift
    result = normalized_x * weight + bias
    
    # Store final result
    tl.store(output_base_ptr + hidden_idx, result, mask=mask)

@torch.fx.wrap
def fused_dropout_layernorm_forward(input_tensor, ln_weight, ln_bias):
    batch_size, seq_len, hidden_size = input_tensor.shape
    
    # Create output tensors
    output = torch.empty_like(input_tensor)
    dropout_output = torch.empty_like(input_tensor)
    
    BLOCK_SIZE = 128  # Power of 2 for better GPU utilization
    
    # Calculate grid dimensions
    batch_grid = (batch_size + 31) // 32  # Process in chunks of 32
    seq_grid = (seq_len + 31) // 32
    hidden_grid = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_dropout_layernorm_kernel[(batch_grid, seq_grid, hidden_grid)](
        input_tensor,
        ln_weight,
        ln_bias,
        output,
        dropout_output,
        hidden_size,
        batch_size,
        seq_len,
        0.1,  # dropout probability
        BLOCK_SIZE,
    )
    
    return dropout_output, output

def replacement_func():
    return fused_dropout_layernorm_forward