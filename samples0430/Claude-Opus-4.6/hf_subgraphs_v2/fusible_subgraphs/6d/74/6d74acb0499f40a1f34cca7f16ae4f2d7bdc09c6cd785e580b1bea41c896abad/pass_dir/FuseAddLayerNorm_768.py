import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
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


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_add_layernorm_kernel(
    bias_ptr,
    weight_ptr,
    x1_ptr,
    x2_ptr,
    out_ptr,
    N: tl.constexpr,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # Load x1 and x2 for this row
    row_start = row_idx * N
    x1 = tl.load(x1_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # Add in native dtype first (matching PyTorch behavior), then cast to float32
    x_sum = x1 + x2
    x = x_sum.to(tl.float32)
    
    # Compute mean and variance using E[X^2] - E[X]^2 formula
    # Since x[>=N]=0.0, sum(x) and sum(x*x) are correct without extra masking
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / N
    sum_x2 = tl.sum(x * x, axis=0)
    var = sum_x2 / N - mean * mean
    
    # Normalize: (x - mean) * rsqrt(var + eps)
    inv_std = tl.math.rsqrt(var + 1e-07)
    normalized = (x - mean) * inv_std
    
    # Load weight and bias, apply affine transformation
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    
    out = weight * normalized + bias
    
    # Store result in float32
    tl.store(out_ptr + row_start + col_offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_layernorm(in_0, in_1, in_2, in_3):
    # in_0: bias [768]
    # in_1: weight [768]
    # in_2: [B, S, 768]
    # in_3: [B, S, 768]
    
    shape = in_2.shape
    N = shape[-1]
    num_rows = in_2.numel() // N
    
    out = torch.empty(shape, dtype=torch.float32, device=in_2.device)
    
    fused_add_layernorm_kernel[(num_rows,)](
        in_0, in_1, in_2, in_3, out,
        N=N,
        num_rows=num_rows,
        BLOCK_SIZE=1024,
        num_warps=8,
        num_stages=2,
    )
    
    return out


def replacement_func():
    return fused_add_layernorm