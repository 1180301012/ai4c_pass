import torch
import triton
import triton.language as tl
import operator


def pattern(in_0, in_1, in_2, in_3, in_4):
    # 3-op fusion: reshape + avg_pool2d + batch_norm (silu stays in remaining graph)
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_S': 64}, num_warps=8, num_stages=4),
    ],
    key=['C', 'S'],
)
@triton.jit
def _fused_avgpool_bn_silu_kernel(
    in4_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, S,
    W_IN: tl.constexpr,
    W_OUT: tl.constexpr,
    eps: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Precompute BN scale and shift once per channel (not per element)
    mean_val = tl.load(mean_ptr   + pid_c).to(tl.float32)
    var_val  = tl.load(var_ptr    + pid_c).to(tl.float32)
    w_val    = tl.load(weight_ptr + pid_c).to(tl.float32)
    b_val    = tl.load(bias_ptr   + pid_c).to(tl.float32)
    scale = w_val * tl.rsqrt(var_val + eps)
    shift = b_val - mean_val * scale  # fused BN: out = avg * scale + shift

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offsets < S
    oh = s_offsets // W_OUT
    ow = s_offsets % W_OUT
    in_base = pid_c * W_IN * W_IN + oh * 2 * W_IN + ow * 2

    v00 = tl.load(in4_ptr + in_base,            mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(in4_ptr + in_base + 1,        mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(in4_ptr + in_base + W_IN,     mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(in4_ptr + in_base + W_IN + 1, mask=mask, other=0.0).to(tl.float32)

    avg = (v00 + v01 + v10 + v11) * 0.25
    bn_out = avg * scale + shift   # 1 mul + 1 add (optimized from 4 mul + 2 add)

    out_offset = pid_c * S + s_offsets
    if IS_BF16:
        tl.store(out_ptr + out_offset, bn_out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + out_offset, bn_out.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_avgpool_bn_silu(in_0, in_1, in_2, in_3, in_4):
    device = in_4.device
    dtype  = in_4.dtype
    C = 512; W_IN = 16; W_OUT = 8; S = W_OUT * W_OUT
    x      = in_4.contiguous()
    mean   = in_0.to(device).contiguous()
    var    = in_1.to(device).contiguous()
    weight = in_3.to(device).contiguous()
    bias   = in_2.to(device).contiguous()
    IS_BF16 = 'bfloat16' in str(dtype)
    out = torch.empty((1, C, W_OUT, W_OUT), dtype=dtype, device=device)
    grid = lambda meta: (C, triton.cdiv(S, meta['BLOCK_S']))
    _fused_avgpool_bn_silu_kernel[grid](
        x, mean, var, weight, bias, out,
        C=C, S=S, W_IN=W_IN, W_OUT=W_OUT, eps=1e-5, IS_BF16=IS_BF16,
    )
    return out


def replacement_func():
    return fused_avgpool_bn_silu