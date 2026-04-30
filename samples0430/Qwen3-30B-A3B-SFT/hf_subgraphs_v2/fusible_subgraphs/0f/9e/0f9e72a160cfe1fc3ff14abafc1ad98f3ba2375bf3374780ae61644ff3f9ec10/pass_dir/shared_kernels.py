"""
Shared Triton kernels + dispatch wrapper used by ALL passes.
Having one shared replacement_func satisfies output_pass_replacement_func_limit=1.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Norm-Div kernel  (L2 normalise each row)
# ---------------------------------------------------------------------------
@triton.jit
def _norm_div_kernel(
    in_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    base = row * N
    x    = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    l    = tl.sqrt(tl.sum(x * x, axis=0))
    tl.store(out_ptr + base + offs, (x / l).to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Exp-Mul kernel  (scalar exp * vector)
# ---------------------------------------------------------------------------
@triton.jit
def _exp_mul_kernel(
    in0_ptr, in2_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs  = tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    scale = tl.load(in0_ptr).to(tl.float32)
    x     = tl.load(in2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (tl.exp(scale) * x).to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Route dispatch wrapper  (single @torch.fx.wrap function shared across passes)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "norm_div":
        in_1 = args[0]
        C          = in_1.shape[-1]
        num_rows   = in_1.numel() // C
        out        = torch.empty_like(in_1)
        _norm_div_kernel[(num_rows,)](in_1, out, N=C, BLOCK_SIZE=512)
        return out
    elif route == "exp_mul":
        in_0 = args[0]
        in_2 = args[1]
        N   = in_2.numel()
        out = torch.empty_like(in_2)
        _exp_mul_kernel[(1,)](in_0, in_2, out, N=N, BLOCK_SIZE=512)
        return out