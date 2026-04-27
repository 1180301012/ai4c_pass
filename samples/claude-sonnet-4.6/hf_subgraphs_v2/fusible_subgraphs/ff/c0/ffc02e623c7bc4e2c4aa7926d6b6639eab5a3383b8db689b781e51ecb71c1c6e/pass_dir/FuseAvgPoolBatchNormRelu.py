import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: adaptive_avg_pool2d → batch_norm (inference)
# relu runs AFTER this subgraph. avg_pool+BN are fused in one Triton pass:
#   for each (b,c): avg = mean(x[b,c,:,:])
#                   out = weight*(avg - mean)/sqrt(var+eps) + bias
# ---------------------------------------------------------------------------
def pattern(in_5, in_1, in_2, in_4, in_3):
    # in_1=running_mean, in_2=running_var, in_4=weight, in_3=bias
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    return tmp_7


def replacement_args(in_5, in_1, in_2, in_4, in_3):
    return (in_5, in_1, in_2, in_4, in_3)


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
def _avg_pool_bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, HW, eps,
    BLOCK_HW: tl.constexpr,
):
    pid  = tl.program_id(0)          # b*C + c
    c    = pid % C
    base = pid * HW
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW
    x    = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    avg  = tl.sum(x, axis=0) / HW
    mean = tl.load(mean_ptr   + c).to(tl.float32)
    var  = tl.load(var_ptr    + c).to(tl.float32)
    w    = tl.load(weight_ptr + c).to(tl.float32)
    b    = tl.load(bias_ptr   + c).to(tl.float32)
    y    = w * (avg - mean) * tl.rsqrt(var + eps) + b
    tl.store(out_ptr + pid, y)


@torch.fx.wrap
def fused_avg_pool_bn(in_5, in_1, in_2, in_4, in_3):
    """
    in_5 : [B, C, H, W]  feature map (before pooling)
    in_1 : [C]  running_mean
    in_2 : [C]  running_var
    in_4 : [C]  weight (gamma)
    in_3 : [C]  bias (beta)
    Returns [B, C, 1, 1]
    """
    B, C, H, W = in_5.shape
    HW  = H * W
    out = torch.empty((B, C, 1, 1), dtype=in_5.dtype, device=in_5.device)
    _avg_pool_bn_kernel[(B * C,)](
        in_5, in_1, in_2, in_4, in_3, out, C, HW, 1e-05,
    )
    return out


def replacement_func():
    return fused_avg_pool_bn