import torch
import triton
import triton.language as tl

# Pattern matching function for LayerNorm
# Matches the exact structure: torch.nn.functional.layer_norm(in_4, (d,), in_1, in_0, 1e-12)
def pattern(in_4, d, in_1, in_0, eps):
    """
    Matches: torch.nn.functional.layer_norm(in_4, (d,), in_1, in_0, 1e-12)
    The pattern must mirror the operations in model.py exactly.
    """
    result = torch.nn.functional.layer_norm(in_4, (d,), in_1, in_0, eps)
    return result

# Extract arguments for replacement
def replacement_args(in_4, d, in_1, in_0, eps):
    return (in_4, d, in_1, in_0, eps)

# Optimized LayerNorm kernel for 3D tensors
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["hidden_size"],
)
@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_rows,
    hidden_size,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm kernel that normalizes along the last dimension.
    Input shape: (n_rows, hidden_size) - flattened view
    Each program handles one row (batch * seq element).
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * hidden_size
    
    # Compute mean across the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_size
    
    # Load row data
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    
    # Compute sum for mean
    sum_val = tl.sum(x_fp32, axis=0)
    mean = sum_val / hidden_size
    
    # Compute variance
    x_centered = x_fp32 - mean
    sum_sq = tl.sum(x_centered * x_centered, axis=0)
    var = sum_sq / hidden_size
    
    # Compute reciprocal std
    rstd = 1.0 / tl.sqrt(var + epsilon)
    
    # Load weight and bias
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    # Apply layer norm
    x_norm = x_centered * rstd
    out = x_norm * w + b
    
    # Store result
    tl.store(output_ptr + row_start + col_offsets, out, mask=mask)

# Optimized wrapper for LayerNorm
@torch.fx.wrap
def layer_norm_wrapper(in_4, d, in_1, in_0, eps):
    """
    Optimized LayerNorm wrapper.
    Matches: torch.nn.functional.layer_norm(in_4, (d,), in_1, in_0, 1e-12)
    """
    # Input shape: (batch, seq_len, hidden_size)
    # Flatten to 2D: (batch * seq_len, hidden_size)
    original_shape = in_4.shape
    batch_seq = original_shape[0] * original_shape[1]
    hidden_size = original_shape[2]
    
    # Create output tensor
    out = torch.empty_like(in_4)
    
    # Flatten input and output for the kernel
    input_flat = in_4.reshape(batch_seq, hidden_size)
    output_flat = out.reshape(batch_seq, hidden_size)
    
    n_rows = batch_seq
    
    # Launch kernel
    layer_norm_kernel[(n_rows,)](
        input_ptr=input_flat,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=output_flat,
        n_rows=n_rows,
        hidden_size=hidden_size,
        epsilon=eps,
    )
    
    return out

def replacement_func():
    return layer_norm_wrapper