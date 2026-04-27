import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Variant: same fusion but with aten.relu_.default (in-place relu)
# ---------------------------------------------------------------------------
def pattern(in_5, in_4, in_3, in_1, in_2):
    tmp_6 = torch.ops.aten.adaptive_avg_pool2d.default(in_5, [1, 1])
    tmp_7 = torch.ops.aten.batch_norm.default(tmp_6, in_4, in_3, in_1, in_2,
                                              False, 0.1, 1e-05, True)
    tmp_8 = torch.ops.aten.relu_.default(tmp_7)
    return tmp_8


def replacement_args(in_5, in_4, in_3, in_1, in_2):
    return (in_5, in_4, in_3, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=1),
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _pool_bn_relu_inplace_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, HW, eps,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    c   = pid % C
    base  = pid * HW
    offs  = tl.arange(0, BLOCK_HW)
    mask  = offs < HW
    x     = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    avg   = tl.sum(x, axis=0) / HW
    mean  = tl.load(mean_ptr   + c).to(tl.float32)
    var   = tl.load(var_ptr    + c).to(tl.float32)
    w     = tl.load(weight_ptr + c).to(tl.float32)
    b_val = tl.load(bias_ptr   + c).to(tl.float32)
    y     = w * (avg - mean) * tl.rsqrt(var + eps) + b_val
    tl.store(out_ptr + pid, tl.maximum(y, 0.0))


@torch.fx.wrap
def fused_bn_relu_inplace(in_5, in_4, in_3, in_1, in_2):
    B, C, H, W = in_5.shape
    HW = H * W
    out = torch.empty((B, C, H, W), dtype=in_5.dtype, device=in_5.device)
    _pool_bn_relu_inplace_kernel[(B * C,)](
        in_5, in_1, in_2, in_4, in_3, out, C, HW, 1e-05,
    )
    return out


def replacement_func():
    return fused_bn_relu_inplace