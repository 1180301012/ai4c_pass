import torch
import triton
import triton.language as tl


# Pre-allocated output cache to avoid torch.empty on every forward call
_output_cache_768 = {}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['num_rows'],
)
@triton.jit
def fused_add_layernorm_768_kernel(
    in2_ptr, in3_ptr,
    weight_ptr, bias_ptr,
    out_add_ptr, out_ln_ptr,
    num_rows,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x2_raw = tl.load(in2_ptr + row_start + offsets, mask=mask, other=0.0)
    x3_raw = tl.load(in3_ptr + row_start + offsets, mask=mask, other=0.0)
    input_dtype = x2_raw.dtype

    x = x2_raw.to(tl.float32) + x3_raw.to(tl.float32)

    # Round to input dtype before storing AND before computing LN stats.
    # This matches what PyTorch's standalone layer_norm reads from memory,
    # ensuring bit-exact results.
    x_rounded = x.to(input_dtype)
    tl.store(out_add_ptr + row_start + offsets, x_rounded, mask=mask)

    x_ln = x_rounded.to(tl.float32)
    mean = tl.sum(x_ln, axis=0) / N
    x_c = x_ln - mean
    x_sq = tl.where(mask, x_c * x_c, 0.0)
    var = tl.sum(x_sq, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

    out_ln = x_c * rstd * w + b
    tl.store(out_ln_ptr + row_start + offsets, out_ln.to(input_dtype), mask=mask)


@torch.fx.wrap
def _fused_add_layernorm_768_impl(in_0, in_1, in_2, in_3):
    N = 768
    num_rows = in_2.numel() // N
    key = (num_rows, str(in_2.dtype))

    # Reuse pre-allocated tensors across forward calls to reduce Python overhead
    if key not in _output_cache_768:
        _output_cache_768[key] = (
            torch.empty(num_rows, N, dtype=in_2.dtype, device=in_2.device),
            torch.empty(num_rows, N, dtype=in_2.dtype, device=in_2.device),
        )
    out_add, out_ln = _output_cache_768[key]

    fused_add_layernorm_768_kernel[(num_rows,)](
        in_2, in_3, in_1, in_0, out_add, out_ln,
        num_rows=num_rows, N=N, eps=1e-5,
    )
    return (out_add, out_ln)


# NOT @torch.fx.wrap — traced by FX to produce two getitem returning nodes.
def fused_add_layernorm_768(in_0, in_1, in_2, in_3):
    result = _fused_add_layernorm_768_impl(in_0, in_1, in_2, in_3)
    return result[0], result[1]


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 768)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-05)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_layernorm_768