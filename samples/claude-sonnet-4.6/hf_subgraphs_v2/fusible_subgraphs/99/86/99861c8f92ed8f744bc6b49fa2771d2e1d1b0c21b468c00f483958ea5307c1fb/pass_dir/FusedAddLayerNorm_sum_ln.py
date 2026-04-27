"""
Pass: FusedAddLayerNorm_sum_ln
Matches: add -> layer_norm, returns (sum, layernorm)
Graphs: Aniemore_unispeech-sat-emotion-russian-resd (float16 and bfloat16)

Uses shared routing dispatch to satisfy output_pass_replacement_func_limit=1.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused element-wise add + layer-norm (N=1024)
# Identical in both pass files so the dispatch wrapper bytecode matches.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=['num_rows'],
)
@triton.jit
def _add_ln_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    sum_ptr, ln_ptr,
    num_rows,
    eps,
    N: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N
    offs = tl.arange(0, N)

    x = tl.load(x_ptr + base + offs).to(tl.float32)
    y = tl.load(y_ptr + base + offs).to(tl.float32)
    z = x + y

    mean = tl.sum(z, axis=0) / N
    d    = z - mean
    var  = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    zn   = d * rstd

    w   = tl.load(w_ptr + offs).to(tl.float32)
    b   = tl.load(b_ptr + offs).to(tl.float32)
    out = zn * w + b

    if IS_BF16:
        tl.store(sum_ptr + base + offs, z.to(tl.bfloat16))
        tl.store(ln_ptr  + base + offs, out.to(tl.bfloat16))
    else:
        tl.store(sum_ptr + base + offs, z.to(tl.float16))
        tl.store(ln_ptr  + base + offs, out.to(tl.float16))


# ---------------------------------------------------------------------------
# Shared routing dispatch wrapper (IDENTICAL in both pass files)
# route="sum_ln" -> (sum, ln)  |  route="ln_sum" -> (ln, sum)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_add_ln_dispatch(in_0, in_1, in_2, in_3, route):
    N = 1024
    num_rows = in_2.numel() // N
    sum_out = torch.empty_like(in_2)
    ln_out  = torch.empty_like(in_2)
    is_bf16 = in_2.dtype == torch.bfloat16
    _add_ln_kernel[(num_rows,)](
        in_2, in_3, in_1, in_0,
        sum_out, ln_out,
        num_rows=num_rows,
        eps=1e-05,
        N=N,
        IS_BF16=is_bf16,
    )
    if route == "sum_ln":
        return (sum_out, ln_out)
    else:
        return (ln_out, sum_out)


# ---------------------------------------------------------------------------
# Pattern / replacement
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return (tmp_2, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "sum_ln")


def replacement_func():
    return _fused_add_ln_dispatch