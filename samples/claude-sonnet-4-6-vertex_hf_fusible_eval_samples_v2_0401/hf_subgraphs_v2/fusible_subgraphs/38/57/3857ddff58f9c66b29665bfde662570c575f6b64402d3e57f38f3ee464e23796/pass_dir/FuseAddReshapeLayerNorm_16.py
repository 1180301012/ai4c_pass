import torch
import triton
import triton.language as tl


# Pre-allocated output cache to avoid torch.empty on every forward call
_output_cache_16 = {}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
    ],
    key=['num_rows'],
)
@triton.jit
def fused_add_layernorm_16_kernel(
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

    # Store the add result in the original dtype
    tl.store(out_add_ptr + row_start + offsets, x.to(input_dtype), mask=mask)

    # Layer norm: mean (masked elements loaded as 0.0 don't affect sum)
    mean = tl.sum(x, axis=0) / N
    x_c = x - mean

    # Variance: zero out padding lanes (BLOCK_SIZE=32 > N=16, so 16 lanes are masked)
    x_sq = tl.where(mask, x_c * x_c, 0.0)
    var = tl.sum(x_sq, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

    out_ln = x_c * rstd * w + b
    tl.store(out_ln_ptr + row_start + offsets, out_ln.to(input_dtype), mask=mask)


@torch.fx.wrap
def _fused_add_layernorm_16_impl(in_0, in_1, in_2, in_3):
    N = 16
    num_rows = in_2.numel() // N
    key = (num_rows, str(in_2.dtype))

    # Reuse pre-allocated tensors across forward calls to reduce Python overhead
    if key not in _output_cache_16:
        _output_cache_16[key] = (
            torch.empty(num_rows, N, dtype=in_2.dtype, device=in_2.device),
            torch.empty(num_rows, N, dtype=in_2.dtype, device=in_2.device),
        )
    out_add, out_ln = _output_cache_16[key]

    fused_add_layernorm_16_kernel[(num_rows,)](
        in_2, in_3, in_1, in_0, out_add, out_ln,
        num_rows=num_rows, N=N, eps=1e-5,
    )
    return (out_add, out_ln)


# NOT @torch.fx.wrap — traced by FX to produce two getitem returning nodes.
def fused_add_layernorm_16(in_0, in_1, in_2, in_3):
    result = _fused_add_layernorm_16_impl(in_0, in_1, in_2, in_3)
    return result[0], result[1]


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2.reshape(-1, 16)
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (16,), in_1, in_0, 1e-05)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_layernorm_16