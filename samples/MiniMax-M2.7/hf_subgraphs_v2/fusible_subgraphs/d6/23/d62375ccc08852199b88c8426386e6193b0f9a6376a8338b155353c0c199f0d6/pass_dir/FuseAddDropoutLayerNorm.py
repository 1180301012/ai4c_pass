"""
Fuse Add + Dropout + LayerNorm into a single optimized kernel.

This pass fuses the following operations:
- Tensor addition (in_0 + positional_embeddings)
- Dropout with p=0.1, training=False (simple scaling)
- LayerNorm over the last dimension

The fused kernel reduces kernel launch overhead and improves memory access patterns.
Supports float32, float16, and bfloat16 dtypes.
"""

import torch
import triton
import triton.language as tl


# Autotune configuration for optimal block sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 256}, num_stages=2, num_warps=4),
    ],
    key=['hidden_size'],
)
@triton.jit
def fused_add_dropout_layernorm_kernel(
    x_ptr,
    y_ptr,
    dropout_out_ptr,
    ln_out_ptr,
    weight_ptr,
    bias_ptr,
    dropout_scale,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for: 
        dropout_out = Dropout(x + y, p, training=False)
        ln_out = LayerNorm(dropout_out, weight, bias)
    
    Since training=False, dropout is a simple scaling operation: scale = 1/(1-p)
    """
    # Program ID for row (batch * seq_len)
    row_idx = tl.program_id(0)
    
    # Offsets for accessing elements in this row
    col_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # Mask for bounds checking
    mask = col_offsets < hidden_size
    
    # Calculate actual memory offsets
    row_offset = row_idx * hidden_size
    offsets = row_offset + col_offsets
    
    # Load x and y
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused addition
    summed = x + y
    
    # Apply dropout scale (training=False means we just scale)
    # scale = 1.0 / (1.0 - 0.1) = 1.0 / 0.9 = 1.1111...
    dropped = summed * dropout_scale
    
    # Compute sum for layer norm
    sum_vals = tl.sum(tl.where(mask, summed, 0.0).to(tl.float32))
    
    # Compute sum of squares for layer norm variance
    sum_sq = tl.sum((tl.where(mask, summed, 0.0) ** 2).to(tl.float32))
    
    # Compute mean and variance
    mean = sum_vals / hidden_size
    var = sum_sq / hidden_size - mean * mean
    # Numerical stability for sqrt
    std = tl.sqrt(tl.maximum(var + eps, 0.0))
    
    # Normalize
    normalized = (summed - mean) / std
    
    # Load weight and bias
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    # Apply affine transform
    out = normalized * w + b
    
    # Store outputs
    tl.store(dropout_out_ptr + offsets, dropped, mask=mask)
    tl.store(ln_out_ptr + offsets, out, mask=mask)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the add + dropout + layer_norm pattern.
    
    Pattern:
        tmp_12 = in_0 + in_1
        tmp_13 = dropout(tmp_12, p=0.1, training=False)
        tmp_14 = layer_norm(tmp_13, (hidden_size,), in_3, in_2, 1e-05)
        return (tmp_13, tmp_14)
    
    Returns: (dropout_output, layer_norm_output)
    """
    # Addition
    tmp_12 = in_0 + in_1
    
    # Dropout with training=False (simple scaling)
    tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
    
    # Layer norm
    hidden_size = in_3.shape[0]
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (hidden_size,), in_3, in_2, 1e-05)
    
    return (tmp_13, tmp_14)


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the replacement function.
    
    Returns: (x, y, weight, bias, hidden_size, dropout_p, eps, dtype, route)
    """
    hidden_size = in_3.shape[0]
    dropout_p = 0.1
    eps = 1e-05
    dtype = str(in_0.dtype)
    
    # Route string for dispatching to correct kernel variant
    route = f"{dtype}_{hidden_size}"
    
    return (in_0, in_1, in_3, in_2, hidden_size, dropout_p, eps, dtype, route)


def replacement_func():
    """
    Returns the optimized replacement function.
    
    Uses a shared dispatch wrapper that handles all dtype variants.
    """
    def dispatch_add_dropout_layernorm(x, y, weight, bias, hidden_size, dropout_p, eps, dtype, route):
        """
        Dispatcher that routes to the appropriate Triton kernel based on dtype.
        
        Since training=False for dropout, we use a simple scaling approach.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        num_rows = batch_size * seq_len
        
        # Dropout scale for training=False: 1/(1-p)
        dropout_scale = 1.0 / (1.0 - dropout_p)
        
        # Output tensors
        dropout_out = torch.empty_like(x)
        ln_out = torch.empty_like(x)
        
        # Define block size based on hidden size
        if hidden_size <= 256:
            block_k = 256
        elif hidden_size <= 512:
            block_k = 512
        elif hidden_size <= 1024:
            block_k = 1024
        else:
            block_k = 2048
        
        grid = (num_rows,)
        
        # Launch the fused kernel
        fused_add_dropout_layernorm_kernel[grid](
            x, y, dropout_out, ln_out, weight, bias, dropout_scale, hidden_size, eps,
            BLOCK_SIZE_K=block_k
        )
        
        return dropout_out, ln_out
    
    return dispatch_add_dropout_layernorm