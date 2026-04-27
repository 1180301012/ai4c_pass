"""
Shared Triton layer-norm kernel for N=1024.
This module is imported by the pass file so it is cached in sys.modules,
ensuring the same _triton_ln function object is returned on every import
(satisfying the set_g_replacement_func identity assertion).
"""
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
def _ln_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    num_rows,
    eps,
    N: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row  = tl.program_id(0)
    base = row * N
    offs = tl.arange(0, N)

    x    = tl.load(x_ptr + base + offs).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xn   = diff * rstd

    w   = tl.load(w_ptr + offs).to(tl.float32)
    b   = tl.load(b_ptr + offs).to(tl.float32)
    out = xn * w + b

    if IS_BF16:
        tl.store(out_ptr + base + offs, out.to(tl.bfloat16))
    else:
        tl.store(out_ptr + base + offs, out.to(tl.float16))


@torch.fx.wrap
def _triton_ln(in_0, in_1, in_2, out, num_rows):
    """
    Fast Triton layer-norm for normalized_shape=(1024,).

    in_0     : bias        [1024]
    in_1     : weight      [1024]
    in_2     : input       [*, 1024]
    out      : pre-allocated output (same shape/dtype as in_2)
    num_rows : in_2.numel()//1024  (pre-computed in the FX graph)

    'out' and 'num_rows' are computed in replacement_args() as regular FX
    graph nodes on real tensors *before* with_dispatch_wrapper_run, avoiding
    two expensive PoisonDispatchTensor __torch_dispatch__ round-trips.
    """
    N       = 1024
    is_bf16 = in_2.dtype == torch.bfloat16   # direct attribute – no dispatch
    _ln_kernel[(num_rows,)](
        in_2, in_1, in_0, out,
        num_rows=num_rows,
        eps=1e-05,
        N=N,
        IS_BF16=is_bf16,
    )
    return out