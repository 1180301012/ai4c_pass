import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Variant C – full 6-arg pattern that mirrors model.py exactly:
#   batch_norm + add + relu(F.relu, inplace=False) + mean(method, keepdim=True)
# ─────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# ─────────────────────────────────────────────────────────────────────────────
# Fused kernel: BN inference + residual add + ReLU + spatial mean
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4,  num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _v3_fused_bn_add_relu_mean_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, res_ptr,
    out_ptr, mean_out_ptr,
    C, HW, eps,
    DTYPE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    c   = pid % C

    bn_mean = tl.load(mean_ptr   + c).to(tl.float32)
    bn_var  = tl.load(var_ptr    + c).to(tl.float32)
    bn_w    = tl.load(weight_ptr + c).to(tl.float32)
    bn_b    = tl.load(bias_ptr   + c).to(tl.float32)

    inv_std = bn_w / tl.sqrt(bn_var + eps)
    shift   = bn_b - bn_mean * inv_std

    base = pid * HW
    acc  = 0.0

    for block_start in range(0, HW, BLOCK_HW):
        idx  = block_start + tl.arange(0, BLOCK_HW)
        mask = idx < HW

        x   = tl.load(x_ptr  + base + idx, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)

        y = tl.maximum(inv_std * x + shift + res, 0.0)
        tl.store(out_ptr + base + idx, y.to(DTYPE), mask=mask)
        y_valid = tl.where(mask, y, tl.zeros_like(y))
        acc = acc + tl.sum(y_valid, axis=0)

    tl.store(mean_out_ptr + pid, (acc / HW).to(DTYPE))


_DTYPE_MAP_V3 = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


@torch.fx.wrap
def v3_fused_bn_add_relu_mean(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0: running_mean [C]
    in_1: running_var  [C]
    in_2: bias (beta)  [C]
    in_3: weight (gamma)[C]
    in_4: BN input     [N, C, H, W]
    in_5: residual     [N, C, H, W]
    """
    N, C, H, W = in_4.shape
    HW = H * W
    out      = torch.empty_like(in_4)
    mean_out = torch.empty((N, C, 1, 1), dtype=in_4.dtype, device=in_4.device)
    DTYPE = _DTYPE_MAP_V3[in_4.dtype]
    _v3_fused_bn_add_relu_mean_kernel[(N * C,)](
        in_4, in_0, in_1, in_3, in_2, in_5,
        out, mean_out.view(N * C),
        C, HW, 1e-05, DTYPE,
    )
    return (out, mean_out)


def replacement_func():
    return v3_fused_bn_add_relu_mean