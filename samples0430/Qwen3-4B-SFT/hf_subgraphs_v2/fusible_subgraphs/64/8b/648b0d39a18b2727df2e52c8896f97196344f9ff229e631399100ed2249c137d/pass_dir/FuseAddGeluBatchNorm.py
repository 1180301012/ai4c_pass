import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the BatchNorm (inference) + trivial 0+no-op subgraph.
    The gelu output (tmp_5) is the pattern's INPUT (in_4).
    Only tmp_7 is returned — a single output satisfies the framework.
    """
    tmp_6 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _bn_trivial_kernel(
    x_ptr,
    running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid: (N * C,)
    Input x = gelu pre-norm tensor (shape [N, C, H, W]).
    Computes: out = bn(x) = scale * x + shift  (scale/shift pre-computed)
    Note: tmp_7 = 0 + bn_output = bn_output (trivial no-op).
    """
    pid = tl.program_id(0)
    c = pid % C

    rm  = tl.load(running_mean_ptr + c).to(tl.float32)
    rv  = tl.load(running_var_ptr  + c).to(tl.float32)
    w   = tl.load(weight_ptr       + c).to(tl.float32)
    b   = tl.load(bias_ptr         + c).to(tl.float32)

    scale = w * tl.rsqrt(rv + eps)
    shift = b - rm * scale

    base = pid * HW
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HW

    x    = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    out  = x * scale + shift
    tl.store(out_ptr + base + offs, out.to(x.dtype), mask=mask)


@torch.fx.wrap
def triton_bn_trivial(in_0, in_1, in_2, in_3, in_4):
    """
    Fused inference-mode BatchNorm + trivial 0+no-op.
    in_0=running_mean, in_1=running_var, in_2=bias, in_3=weight, in_4=input=[N,C,H,W]
    Returns: bn_result (= tmp_7 since tmp_7 = 0 + bn_result)
    """
    N, C, H, W = in_4.shape
    HW = H * W
    NC = N * C

    out = torch.empty_like(in_4)

    _bn_trivial_kernel[(NC,)](
        in_4,
        in_0, in_1,   # running_mean, running_var
        in_3, in_2,   # weight, bias
        out,
        C, HW,
        eps=1e-5,
    )
    return out


def replacement_func():
    return triton_bn_trivial