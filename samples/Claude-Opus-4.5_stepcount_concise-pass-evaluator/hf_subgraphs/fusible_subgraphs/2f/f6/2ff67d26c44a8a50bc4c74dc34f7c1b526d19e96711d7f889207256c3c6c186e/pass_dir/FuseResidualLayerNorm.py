import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation in model.py
def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern for fused residual + LayerNorm
    in_0: bias [768]
    in_1: weight [768]
    in_2: residual input [B, S, 768]
    in_3: hidden states [B, S, 768]
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3 + tmp_2
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
    tmp_14 = tmp_1 * tmp_13
    tmp_15 = tmp_14 + tmp_0
    return (tmp_15,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Compile-time constants for hidden_size=768
HIDDEN_SIZE_CONST: tl.constexpr = 768
BLOCK_SIZE_CONST: tl.constexpr = 1024
EPS_CONST: tl.constexpr = 1e-07
INV_HIDDEN: tl.constexpr = 1.0 / 768


@triton.jit
def fused_residual_layernorm_kernel(
    x_ptr, residual_ptr, weight_ptr, bias_ptr, output_ptr, stride,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride
    col_offsets = tl.arange(0, BLOCK_SIZE_CONST)
    mask = col_offsets < HIDDEN_SIZE_CONST
    
    # Load data with coalesced access
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # Fused add + LayerNorm
    hidden_f32 = (x + residual).to(tl.float32)
    hidden_masked = tl.where(mask, hidden_f32, 0.0)
    mean = tl.sum(hidden_masked, axis=0) * INV_HIDDEN
    
    centered = hidden_f32 - mean
    centered_sq_masked = tl.where(mask, centered * centered, 0.0)
    var = tl.sum(centered_sq_masked, axis=0) * INV_HIDDEN
    
    rstd = tl.rsqrt(var + EPS_CONST)
    normalized = centered * rstd
    
    # Load weight/bias and apply
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    output = weight * normalized + bias
    
    tl.store(output_ptr + row_start + col_offsets, output.to(tl.float32), mask=mask)


@torch.fx.wrap
def fused_residual_layernorm(bias, weight, residual, hidden_states):
    """Fused residual + LayerNorm operation"""
    orig_shape = hidden_states.shape
    hidden_size = orig_shape[-1]
    num_rows = hidden_states.numel() // hidden_size
    
    hidden_flat = hidden_states.contiguous().view(-1, hidden_size)
    residual_flat = residual.contiguous().view(-1, hidden_size)
    output_flat = torch.empty_like(hidden_flat)
    
    # Optimal config: 4 warps, 2 stages for best performance
    fused_residual_layernorm_kernel[(num_rows,)](
        hidden_flat, residual_flat, weight, bias, output_flat, hidden_size,
        num_warps=4, num_stages=2,
    )
    
    return output_flat.view(orig_shape)


def replacement_func():
    return fused_residual_layernorm