import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_4 / in_3
    tmp_4 = tmp_3.to(torch.float32)
    tmp_5 = tmp_0.unsqueeze(-1)
    tmp_6 = tmp_4 * tmp_5
    tmp_7 = tmp_6.to(torch.float32)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (320,), tmp_2, tmp_1, 1e-05)
    return tmp_7, tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_div_mul_layernorm_kernel(
    in_4_ptr,
    in_3_ptr,
    in_0_ptr,
    in_2_ptr,
    in_1_ptr,
    out_7_ptr,
    out_8_ptr,
    S,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one position in batch x sequence)
    row_idx = tl.program_id(0)
    
    b_idx = row_idx // S
    s_idx = row_idx % S
    
    # Load normalization factor for this batch (in_3 has shape [B, 1, 1])
    norm_factor = tl.load(in_3_ptr + b_idx)
    
    # Load attention mask and convert to float32 (in_0 has shape [B, S])
    mask_val = tl.load(in_0_ptr + b_idx * S + s_idx).to(tl.float32)
    
    # Column offsets for loading/storing
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < D
    
    # Calculate row start position in flattened tensor
    row_start = row_idx * D
    
    # Load input row from in_4
    x = tl.load(in_4_ptr + row_start + col_offsets, mask=col_mask, other=0.0)
    
    # Division by normalization factor and multiply by mask
    x = (x / norm_factor) * mask_val
    
    # Store intermediate result (tmp_7)
    tl.store(out_7_ptr + row_start + col_offsets, x, mask=col_mask)
    
    # === LayerNorm computation using single-pass for mean and variance ===
    # Compute sum(x) and sum(x^2) in one pass
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    
    # Compute mean and variance
    mean = sum_x / D
    var = sum_x2 / D - mean * mean
    
    # Compute normalized values: (x - mean) / sqrt(var + eps)
    rstd = 1.0 / tl.sqrt(var + EPS)
    x_norm = tl.where(col_mask, (x - mean) * rstd, 0.0)
    
    # Load LayerNorm weight (in_2) and bias (in_1)
    weight = tl.load(in_2_ptr + col_offsets, mask=col_mask, other=0.0)
    bias = tl.load(in_1_ptr + col_offsets, mask=col_mask, other=0.0)
    
    # Apply affine transformation: weight * x_norm + bias
    out = weight * x_norm + bias
    
    # Store final result (tmp_8)
    tl.store(out_8_ptr + row_start + col_offsets, out, mask=col_mask)


@torch.fx.wrap
def _fused_kernel_call(in_0, in_1, in_2, in_3, in_4):
    B, S, D = in_4.shape
    
    # Allocate output tensors
    out_7 = torch.empty((B, S, D), dtype=torch.float32, device=in_4.device)
    out_8 = torch.empty((B, S, D), dtype=torch.float32, device=in_4.device)
    
    # Launch kernel - one program per row
    num_rows = B * S
    BLOCK_SIZE = 512  # Power of 2 >= D (320)
    
    fused_div_mul_layernorm_kernel[(num_rows,)](
        in_4, in_3, in_0, in_2, in_1,
        out_7, out_8,
        S,
        D,
        1e-05,
        BLOCK_SIZE,
        num_warps=4,
    )
    
    return (out_7, out_8)


def fused_div_mul_layernorm(in_0, in_1, in_2, in_3, in_4):
    result = _fused_kernel_call(in_0, in_1, in_2, in_3, in_4)
    return result[0], result[1]


def replacement_func():
    return fused_div_mul_layernorm