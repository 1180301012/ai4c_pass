import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 16}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 32}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 1}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4}),
    ],
    key=['N'],
)
@triton.jit
def _fused_add_ln_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr, out_ptr,
    N, D,
    OUT_DTYPE: tl.constexpr,   # 0=fp32, 1=fp16, 2=bf16
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: (x + y).float32 layer-norm → weight * x_hat + bias, stored back in original dtype."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D
    base = row * D

    # ---- load inputs, upcast to float32 for precision ----
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    s = x + y

    # ---- mean (padded slots loaded as 0, don't affect sum) ----
    mean_val = tl.sum(s, axis=0) / D

    # ---- center, zero padding slots explicitly ----
    centered = tl.where(mask, s - mean_val, tl.zeros_like(s))

    # ---- variance ----
    var_val = tl.sum(centered * centered, axis=0) / D

    # ---- normalize ----
    eps = 1e-7
    norm = centered / tl.sqrt(var_val + eps)

    # ---- sign-clamp: original code uses abs(rms) numerator ----
    rms = tl.sqrt(var_val + eps)
    x_hat = tl.where(norm >= 0.0, norm, -norm)

    # ---- load weight/bias, scale+shift in float32 ----
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out_v = w * x_hat + b

    # ---- cast back to original dtype and store ----
    if OUT_DTYPE == 1:          # float16
        tl.store(out_ptr + base + offs, out_v.to(tl.float16), mask=mask)
    elif OUT_DTYPE == 2:        # bfloat16
        tl.store(out_ptr + base + offs, out_v.to(tl.bfloat16), mask=mask)
    else:                       # float32
        tl.store(out_ptr + base + offs, out_v, mask=mask)


@torch.fx.wrap
def fused_add_layernorm(in_0, in_1, in_2, in_3):
    """
    Replacement for: (in_3 + in_2).float() → LayerNorm → in_1 * norm + in_0
    in_0 : bias   [D], any float dtype
    in_1 : weight [D], any float dtype
    in_2 : [*, D],    any float dtype
    in_3 : [*, D],    any float dtype
    returns: [*, D],  same dtype as in_2 / in_3
    """
    D = in_2.shape[-1]
    N = in_2.numel() // D

    w_1 = in_1 if in_1.is_contiguous() else in_1.contiguous()
    b_0 = in_0 if in_0.is_contiguous() else in_0.contiguous()

    out = torch.empty_like(in_2)

    if in_2.dtype == torch.float16:
        out_dtype = 1   # float16
    elif in_2.dtype == torch.bfloat16:
        out_dtype = 2   # bfloat16
    else:
        out_dtype = 0   # float32

    _fused_add_ln_kernel[(N,)](
        in_3, in_2, w_1, b_0, out,
        N, D,
        OUT_DTYPE=out_dtype,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

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


def replacement_func():
    return fused_add_layernorm