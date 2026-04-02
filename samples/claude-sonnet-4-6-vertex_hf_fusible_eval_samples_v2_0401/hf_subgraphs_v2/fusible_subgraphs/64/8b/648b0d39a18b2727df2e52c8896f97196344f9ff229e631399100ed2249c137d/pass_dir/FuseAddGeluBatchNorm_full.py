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
def fused_add_gelu_bn_kernel(
    in4_ptr, in5_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    gelu_out_ptr, bn_out_ptr,
    C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Full fusion: add(in4, in5) + GELU + BN-inference in one kernel.
    3D grid: (N, C, ceil(HW / BLOCK_SIZE))
    Reads: in4 + in5 (2 tensors).  Writes: gelu_out + bn_out (2 tensors).
    Saves the iadd write + gelu re-read vs the 5-arg fused variant.
    """
    pid_n  = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Load BN params for this channel (scalars)
    mean = tl.load(mean_ptr   + pid_c).to(tl.float32)
    var  = tl.load(var_ptr    + pid_c).to(tl.float32)
    w    = tl.load(weight_ptr + pid_c).to(tl.float32)
    b    = tl.load(bias_ptr   + pid_c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    scale   = w * inv_std
    shift   = b - mean * scale

    base_offset = pid_n * C * HW + pid_c * HW
    hw_offs = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = hw_offs < HW
    flat_offs = base_offset + hw_offs

    # Load BOTH input tensors
    xv4 = tl.load(in4_ptr + flat_offs, mask=mask, other=0.0)
    xv5 = tl.load(in5_ptr + flat_offs, mask=mask, other=0.0)

    # 1) Add
    x = xv4.to(tl.float32) + xv5.to(tl.float32)

    # 2) Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))

    # 3) BN inference: bn = gelu * scale + shift
    bn = gelu * scale + shift

    tl.store(gelu_out_ptr + flat_offs, gelu.to(xv4.dtype), mask=mask)
    tl.store(bn_out_ptr   + flat_offs, bn.to(xv4.dtype),   mask=mask)


@torch.fx.wrap
def _run_fused_add_gelu_bn(running_mean, running_var, bias, weight, in4, in5):
    """
    Full-fusion kernel launcher: add + GELU + BN in a single pass.
    """
    N, C, H, W = in4.shape
    HW  = H * W
    eps = 1e-5

    gelu_out = torch.empty_like(in4)
    bn_out   = torch.empty_like(in4)

    grid = lambda meta: (N, C, triton.cdiv(HW, meta['BLOCK_SIZE']))

    fused_add_gelu_bn_kernel[grid](
        in4, in5,
        running_mean, running_var, weight, bias,
        gelu_out, bn_out,
        C, HW,
        eps,
    )

    return gelu_out, bn_out


# FX-traceable wrapper (NOT @torch.fx.wrap): produces 2 getitem nodes.
def fused_add_gelu_bn(running_mean, running_var, bias, weight, in4, in5):
    result = _run_fused_add_gelu_bn(running_mean, running_var, bias, weight, in4, in5)
    return result[0], result[1]


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Full pattern: add + gelu + batch_norm + 0+identity.

    Uses explicit non-in-place `in_4 + in_5` so FX symbolic_trace creates
    a regular `add` node (same as the model's `in_4 += in_5` which also
    falls back to `add` via Proxy.__add__).

    Argument mapping:
      in_0 = running_mean [C]
      in_1 = running_var  [C]
      in_2 = bias         [C]
      in_3 = weight       [C]
      in_4 = tensor A     [N,C,H,W]
      in_5 = tensor B     [N,C,H,W]
    """
    in_6 = in_4 + in_5
    tmp_5 = torch.nn.functional.gelu(in_6, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return (tmp_5, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_add_gelu_bn