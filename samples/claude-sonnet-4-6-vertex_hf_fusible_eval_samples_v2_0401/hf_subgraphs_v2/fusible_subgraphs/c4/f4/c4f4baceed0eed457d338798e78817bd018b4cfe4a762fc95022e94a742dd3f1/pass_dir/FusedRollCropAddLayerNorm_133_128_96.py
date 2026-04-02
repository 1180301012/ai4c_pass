"""
Fused pass: roll(shifts=(3,3), dims=(1,2)) + crop[:,:128,:128,:] + add + layer_norm
for the shape variant: in_3 view(-1, 133, 133, 96), output [1, 16384, 96]
"""
import torch
import triton
import triton.language as tl

# ── Triton kernel ─────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _fused_roll_crop_add_ln_133_128_96(
    in2_ptr,       # [1, N, C]  – residual
    in3_ptr,       # [1, H*H, C] – in_3 (contiguous, viewed as flat HxH rows)
    out_add_ptr,   # [1, N, C]
    out_ln_ptr,    # [1, N, C]
    weight_ptr,    # [C]
    bias_ptr,      # [C]
    N,
    INPUT_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Shape constants
    H: tl.constexpr = 133
    H_out: tl.constexpr = 128
    C: tl.constexpr = 96
    eps: tl.constexpr = 1e-5

    n = tl.program_id(0)

    # 2-D position in the H_out × H_out output grid
    i = n // H_out
    j = n % H_out

    # Rolled source position: roll by (3,3) means output[i,j] = input[(i-3)%H, (j-3)%H]
    si = (i - 3 + H) % H
    sj = (j - 3 + H) % H
    src_n = si * H + sj          # flat row index into [H*H, C]

    offsets = tl.arange(0, BLOCK_C)
    mask = offsets < C

    # Load & upcast to fp32 for numerics
    in2 = tl.load(in2_ptr + n * C + offsets, mask=mask, other=0.0).to(tl.float32)
    in3 = tl.load(in3_ptr + src_n * C + offsets, mask=mask, other=0.0).to(tl.float32)

    x = in2 + in3   # residual add

    # Store the sum (tmp_8)
    tl.store(out_add_ptr + n * C + offsets, x.to(INPUT_DTYPE), mask=mask)

    # ── Layer Norm ──────────────────────────────────────────────────────────
    # Mean  (padded lanes are 0, so sum / C is correct)
    mean = tl.sum(x, axis=0) / C

    # Variance (mask out padded lanes before squaring)
    diff = x - mean
    diff_sq = tl.where(mask, diff * diff, 0.0)
    var = tl.sum(diff_sq, axis=0) / C

    rstd = 1.0 / tl.sqrt(var + eps)
    norm = diff * rstd

    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias   = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

    out_ln = norm * weight + bias
    tl.store(out_ln_ptr + n * C + offsets, out_ln.to(INPUT_DTYPE), mask=mask)


# ── dtype helpers ──────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


# ── Host wrapper ───────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_roll_crop_add_ln_133_128_96(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [96]
    in_1 : weight [96]
    in_2 : residual [1, 16384, 96]
    in_3 : permuted [1, 19, 7, 19, 7, 96]  (or any shape with 133*133*96 elements per batch)
    Returns (sum, layer_norm_output) both [1, 16384, 96]
    """
    H, H_out, C = 133, 128, 96
    N = H_out * H_out  # 16384

    in3_flat = in_3.contiguous().view(1, H * H, C)   # [1, 17689, 96]

    out_add = torch.empty_like(in_2)
    out_ln  = torch.empty_like(in_2)

    INPUT_DTYPE = _DTYPE_MAP[in_2.dtype]

    _fused_roll_crop_add_ln_133_128_96[(N,)](
        in_2, in3_flat, out_add, out_ln,
        in_1, in_0,     # weight, bias
        N,
        INPUT_DTYPE=INPUT_DTYPE,
    )

    return out_add, out_ln


# ── Pattern / replacement API ─────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 133, 133, 96)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 16384, 96)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), in_1, in_0, 1e-05)
    return (tmp_8, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_roll_crop_add_ln_133_128_96