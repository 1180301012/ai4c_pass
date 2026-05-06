import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: in_1 += in_0  →  .float()  →  softmax(dim=-1)  →  .type_as(in_2)
#          →  dropout(training=False)   [which is identity]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    in_1 += in_0
    in_2 = in_1
    tmp_1 = in_2.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(in_2)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused add + softmax in fp32 + cast-back to original dtype
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32},  num_warps=1),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_cols'],
)
@triton.jit
def _fused_add_softmax_kernel(
    in0_ptr, in1_ptr, out_ptr,
    n_cols,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load both inputs (native dtype)
    x = tl.load(in0_ptr + row * n_cols + offsets, mask=mask, other=0.0)
    y = tl.load(in1_ptr + row * n_cols + offsets, mask=mask, other=0.0)

    # Upcast to fp32, compute element-wise sum
    xy_f32 = x.to(tl.float32) + y.to(tl.float32)

    # --- Numerically-stable softmax ---
    x_max   = tl.max(xy_f32, axis=0)
    x_shift = tl.where(mask, xy_f32 - x_max, -float('inf'))
    x_exp   = tl.exp(x_shift)
    x_sum   = tl.sum(x_exp, axis=0)
    x_out   = x_exp / x_sum

    # Cast result back to original dtype
    if IS_FP16:
        result = x_out.to(tl.float16)
    elif IS_BF16:
        result = x_out.to(tl.bfloat16)
    else:
        result = x_out          # float32 in

    tl.store(out_ptr + row * n_cols + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper returned by replacement_func()
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    orig_dtype  = in_0.dtype
    n_cols      = in_0.numel() // in_0.shape[0]   # last-dim size (= S*S for [B,H,S,S])
    n_rows      = in_0.numel() // n_cols          # number of softmax rows

    out = torch.empty_like(in_0)

    # Call autotuned kernel directly using bracket notation.
    # IS_FP16/IS_BF16 trigger per-dtype specialisation at compile time.
    _fused_add_softmax_kernel[(n_rows,)](
        out, in_0, in_1,
        n_cols,
        IS_FP16=(orig_dtype == torch.float16),
        IS_BF16=(orig_dtype == torch.bfloat16),
    )

    return out


def replacement_func():
    return fused_add_softmax