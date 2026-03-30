import torch
import triton
import triton.language as tl



@triton.jit
def fused_layernorm_dropout_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, seq_len, embed_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm + Dropout kernel
    """
    # Each program handles one element
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < batch_size * seq_len * embed_dim
    
    # Calculate indices
    batch_seq_idx = offset // embed_dim
    embed_idx = offset % embed_dim
    
    # Load input and parameters
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + embed_idx, mask=embed_idx < embed_dim, other=1.0)
    bias = tl.load(bias_ptr + embed_idx, mask=embed_idx < embed_dim, other=0.0)
    
    # Compute mean and variance for this row
    row_offset = batch_seq_idx * embed_dim
    row_ptr = x_ptr + row_offset
    
    # Simple mean calculation (in practice, you'd want a more complex reduction)
    row_vals = tl.load(row_ptr + tl.arange(0, embed_dim), mask=tl.arange(0, embed_dim) < embed_dim)
    row_mean = tl.sum(row_vals) / embed_dim
    row_var = tl.sum((row_vals - row_mean) * (row_vals - row_mean)) / embed_dim
    
    # LayerNorm
    norm_x = (x - tl.full_like(x, row_mean, x.dtype)) / tl.sqrt(tl.full_like(x, row_var, x.dtype) + eps)
    norm_x = norm_x * weight + bias
    
    # Dropout (always keep for training=False)
    dropout_x = norm_x * (0.9)  # p=0.1, so keep probability is 0.9
    
    # Store output
    tl.store(out_ptr + offset, dropout_x, mask=mask)

@torch.fx.wrap
def fused_layernorm_dropout(x, weight, bias, eps):
    batch_size, seq_len, embed_dim = x.shape
    
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    num_programs = (batch_size * seq_len * embed_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_layernorm_dropout_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    def fused_layernorm_dropout_wrapper(x, weight, bias, eps):
        # Compute LayerNorm first (required to return this value)
        normalized = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        
        # Use fused kernel for dropout
        dropout_result = fused_layernorm_dropout(x, weight, bias, eps)
        
        return normalized, dropout_result
    
    return fused_layernorm_dropout_wrapper