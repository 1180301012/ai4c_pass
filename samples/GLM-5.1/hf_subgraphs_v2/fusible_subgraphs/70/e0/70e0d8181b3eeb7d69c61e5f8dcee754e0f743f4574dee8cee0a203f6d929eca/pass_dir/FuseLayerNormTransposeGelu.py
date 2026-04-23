import torch
import triton
import triton.language as tl
import math

# Pattern matching function - must match the exact computation from model.py
def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Fused kernel: layer_norm + transpose + gelu
# Input: [batch, rows, cols] where cols=512 (normalized dimension)
# Output: [batch, cols, rows] - transposed with GELU applied

@triton.jit
def fused_layernorm_transpose_gelu_kernel(
    input_ptr,      # [batch, rows, cols]
    weight_ptr,     # [cols]
    bias_ptr,       # [cols]
    output_ptr,     # [batch, cols, rows] - transposed output
    batch_size,
    rows,           # number of rows (e.g., 3999)
    cols,           # normalized dimension (e.g., 512)
    eps,            # 1e-05
    input_batch_stride,
    input_row_stride,
    input_col_stride,
    output_batch_stride,
    output_col_stride,   # stride for the "cols" dimension in output (which is now rows in original)
    output_row_stride,   # stride for the "rows" dimension in output (which is now cols in original)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row of input
    row_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    # Pointer to the start of this row in input
    row_start = input_ptr + batch_idx * input_batch_stride + row_idx * input_row_stride
    
    # Load the entire row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < cols
    
    # Load input row
    x = tl.load(row_start + col_offsets * input_col_stride, mask=mask, other=0.0).to(tl.float32)
    
    # Load weight and bias
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute layer norm: mean
    mean = tl.sum(x, axis=0) / cols
    
    # Compute variance
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / cols
    
    # Normalize
    x_normed = x_centered / tl.sqrt(variance + eps)
    
    # Apply weight and bias
    result = x_normed * w + b
    
    # Apply GELU using exact erf formula: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt_2 = 1.4142135623730951
    result_gelu = result * 0.5 * (1.0 + tl.erf(result / sqrt_2))
    
    # Write to transposed output: output[batch, col, row]
    # For each column j in the input row, write to output[batch, j, row_idx]
    output_row_start = output_ptr + batch_idx * output_batch_stride
    
    # Scatter write: for col j, write to output position [batch, j, row_idx]
    # output[batch, j, row_idx] = output_ptr + batch * output_batch_stride + j * output_col_stride + row_idx * output_row_stride
    scatter_offsets = col_offsets * output_col_stride + row_idx * output_row_stride
    tl.store(output_row_start + scatter_offsets, result_gelu.to(input_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['cols'],
)
@triton.jit
def fused_layernorm_transpose_gelu_kernel_autotuned(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    rows,
    cols,
    eps,
    input_batch_stride,
    input_row_stride,
    input_col_stride,
    output_batch_stride,
    output_col_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row of input
    row_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    # Pointer to the start of this row in input
    row_start = input_ptr + batch_idx * input_batch_stride + row_idx * input_row_stride
    
    # Load the entire row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < cols
    
    # Load input row
    x = tl.load(row_start + col_offsets * input_col_stride, mask=mask, other=0.0).to(tl.float32)
    
    # Load weight and bias
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute layer norm: mean
    mean = tl.sum(x, axis=0) / cols
    
    # Compute variance
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / cols
    
    # Normalize
    x_normed = x_centered / tl.sqrt(variance + eps)
    
    # Apply weight and bias
    result = x_normed * w + b
    
    # Apply GELU using exact erf formula: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt_2 = 1.4142135623730951
    result_gelu = result * 0.5 * (1.0 + tl.erf(result / sqrt_2))
    
    # Write to transposed output
    output_row_start = output_ptr + batch_idx * output_batch_stride
    scatter_offsets = col_offsets * output_col_stride + row_idx * output_row_stride
    tl.store(output_row_start + scatter_offsets, result_gelu.to(input_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_layernorm_transpose_gelu(in_0, in_1, in_2):
    # in_0: bias [512]
    # in_1: weight [512]
    # in_2: input [batch, rows, 512]
    
    input_tensor = in_2
    weight = in_1
    bias = in_0
    
    batch_size = input_tensor.shape[0]
    rows = input_tensor.shape[1]
    cols = input_tensor.shape[2]  # 512
    
    # Output shape: [batch, cols, rows] = [batch, 512, rows]
    output = torch.empty((batch_size, cols, rows), dtype=input_tensor.dtype, device=input_tensor.device)
    
    eps = 1e-05
    
    # Grid: one program per row, per batch
    grid = (rows, batch_size)
    
    fused_layernorm_transpose_gelu_kernel_autotuned[grid](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        rows=rows,
        cols=cols,
        eps=eps,
        input_batch_stride=input_tensor.stride(0),
        input_row_stride=input_tensor.stride(1),
        input_col_stride=input_tensor.stride(2),
        output_batch_stride=output.stride(0),
        output_col_stride=output.stride(1),
        output_row_stride=output.stride(2),
    )
    
    return (output,)


def replacement_func():
    return fused_layernorm_transpose_gelu