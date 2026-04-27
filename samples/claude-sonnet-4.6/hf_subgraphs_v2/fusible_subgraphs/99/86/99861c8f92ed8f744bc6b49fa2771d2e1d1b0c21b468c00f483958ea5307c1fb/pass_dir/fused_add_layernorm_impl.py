import torch
import triton
import triton.language as tl


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
def _fused_add_layernorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    sum_out_ptr, ln_out_ptr,
    num_rows,
    eps,
    N: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, N)

    # Load inputs and cast to float32 for numerically-stable computation
    x = tl.load(x_ptr + row_start + offsets).to(tl.float32)
    y = tl.load(y_ptr + row_start + offsets).to(tl.float32)
    z = x + y   # sum result (tmp_2)

    # ---- layer-norm statistics ----
    mean = tl.sum(z, axis=0) / N
    diff = z - mean
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    z_norm = diff * rstd

    # Scale and shift
    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    bias   = tl.load(bias_ptr   + offsets).to(tl.float32)
    out = z_norm * weight + bias

    # Store back with the input's native precision
    if IS_BF16:
        tl.store(sum_out_ptr + row_start + offsets, z.to(tl.bfloat16))
        tl.store(ln_out_ptr  + row_start + offsets, out.to(tl.bfloat16))
    else:
        tl.store(sum_out_ptr + row_start + offsets, z.to(tl.float16))
        tl.store(ln_out_ptr  + row_start + offsets, out.to(tl.float16))


def fused_add_layernorm_compute(in_0, in_1, in_2, in_3):
    """
    Fused elementwise-add + layer-norm.

    Parameters
    ----------
    in_0 : bias,   shape [1024]
    in_1 : weight, shape [1024]
    in_2 : x,      shape [*, 1024]
    in_3 : y,      shape [*, 1024]

    Returns
    -------
    (sum_out, ln_out) both with the same shape and dtype as in_2
    """
    N = 1024
    num_rows = in_2.numel() // N

    sum_out = torch.empty_like(in_2)
    ln_out  = torch.empty_like(in_2)

    is_bf16 = (in_2.dtype == torch.bfloat16)

    _fused_add_layernorm_kernel[(num_rows,)](
        in_2, in_3, in_1, in_0,
        sum_out, ln_out,
        num_rows=num_rows,
        eps=1e-05,
        N=N,
        IS_BF16=is_bf16,
    )

    return sum_out, ln_out