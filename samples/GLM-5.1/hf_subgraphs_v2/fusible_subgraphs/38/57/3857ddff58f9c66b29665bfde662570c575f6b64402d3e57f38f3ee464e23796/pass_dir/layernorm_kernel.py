import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_rows, n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    row_start = row_idx * n_cols
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + n_cols
    mask_float = mask.to(tl.float32)
    
    # Load input in float32 for numerical stability
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean - only count valid elements
    mean = tl.sum(x * mask_float, axis=0) / n_cols
    
    # Compute variance - only count valid elements
    diff = (x - mean) * mask_float
    diff_sq = diff * diff
    var = tl.sum(diff_sq, axis=0) / n_cols
    
    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = diff * rstd
    
    # Load weight and bias in float32
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
    
    result = normalized * w + b
    
    tl.store(output_ptr + offsets, result.to(INPUT_DTYPE), mask=mask)


@torch.fx.wrap
def triton_layernorm(in_0, in_1, tmp_3):
    n_cols = in_0.shape[0]
    n_rows = tmp_3.numel() // n_cols
    
    output = torch.empty((n_rows, n_cols), dtype=tmp_3.dtype, device=tmp_3.device)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    eps = 1e-05
    grid = (n_rows,)
    
    if tmp_3.dtype == torch.float16:
        INPUT_DTYPE = tl.float16
    elif tmp_3.dtype == torch.bfloat16:
        INPUT_DTYPE = tl.bfloat16
    else:
        INPUT_DTYPE = tl.float32
    
    layernorm_kernel[grid](
        tmp_3, in_1, in_0, output,
        n_rows, n_cols,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        INPUT_DTYPE=INPUT_DTYPE,
    )
    
    return output