import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_gelu_bn_kernel(
    x_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    gelu_out_ptr, bn_out_ptr,
    C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    3D grid: (N, C, ceil(HW / BLOCK_SIZE))
    Each program handles BLOCK_SIZE spatial elements for one (batch, channel).
    BN stats are loaded once per program as scalars, then fused into scale/shift.
    """
    pid_n  = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Load BN params for this channel (4 scalar loads)
    mean = tl.load(mean_ptr   + pid_c).to(tl.float32)
    var  = tl.load(var_ptr    + pid_c).to(tl.float32)
    w    = tl.load(weight_ptr + pid_c).to(tl.float32)
    b    = tl.load(bias_ptr   + pid_c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    # Fuse into 2-FMA BN: bn = gelu * scale + shift
    scale = w * inv_std
    shift = b - mean * scale

    base_offset = pid_n * C * HW + pid_c * HW
    hw_offs = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = hw_offs < HW
    flat_offs = base_offset + hw_offs

    xv    = tl.load(x_ptr + flat_offs, mask=mask, other=0.0)
    x_f32 = xv.to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))

    # BN inference: bn = gelu * scale + shift
    bn = gelu * scale + shift

    tl.store(gelu_out_ptr + flat_offs, gelu.to(xv.dtype), mask=mask)
    tl.store(bn_out_ptr   + flat_offs, bn.to(xv.dtype),   mask=mask)


@torch.fx.wrap
def _run_fused_gelu_bn(running_mean, running_var, bias, weight, x):
    """
    Opaque kernel launcher (wrapped so FX does NOT trace inside).
    Fuses GELU + BN-inference into a single kernel pass.
    """
    N, C, H, W = x.shape
    HW  = H * W
    eps = 1e-5

    gelu_out = torch.empty_like(x)
    bn_out   = torch.empty_like(x)

    grid = lambda meta: (N, C, triton.cdiv(HW, meta['BLOCK_SIZE']))

    fused_gelu_bn_kernel[grid](
        x,
        running_mean, running_var, weight, bias,
        gelu_out, bn_out,
        C, HW,
        eps,
    )

    return gelu_out, bn_out


# NOT decorated with @torch.fx.wrap so FX CAN trace into this function.
# FX then sees two explicit getitem nodes → 2 copied_returning_nodes,
# matching the pattern's 2 returning nodes (tmp_5 and tmp_7).
def fused_gelu_bn(running_mean, running_var, bias, weight, x):
    result = _run_fused_gelu_bn(running_mean, running_var, bias, weight, x)
    return result[0], result[1]


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, x):
    """
    Matches the gelu → batch_norm → 0+identity subgraph from model.py.

    The upstream in_4 += in_5 (iadd_) is NOT included here because dynamo
    traces it as 'add_' while FX symbolic_trace of the pattern converts
    'in_4 += in_5' to a regular 'add' — the targets would differ and the
    match would fail.  Instead, 'x' is a wildcard placeholder that matches
    the output of whatever produced the gelu input (the iadd_ result).

    Argument mapping (from weight_meta / model.py):
      in_0 = running_mean  [C]
      in_1 = running_var   [C]
      in_2 = bias          [C]
      in_3 = weight        [C]
      x    = gelu input    [N,C,H,W]  (= result of in_4 += in_5)
    """
    tmp_5 = torch.nn.functional.gelu(x, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return (tmp_5, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, x):
    return (in_0, in_1, in_2, in_3, x)


def replacement_func():
    return fused_gelu_bn