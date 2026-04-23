import torch
import triton
import triton.language as tl


def pattern(running_mean, running_var, bias, weight, x0, x1, x2, x3):
    cat = torch.cat([x0, x1, x2, x3], 1)
    bn = torch.nn.functional.batch_norm(cat, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    silu = torch.nn.functional.silu(bn, inplace=True)
    mean = silu.mean((2, 3), keepdim=True)
    return silu, mean


def replacement_args(running_mean, running_var, bias, weight, x0, x1, x2, x3):
    return running_mean, running_var, bias, weight, x0, x1, x2, x3


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 8, "BLOCK_S": 64}, num_warps=4),
        triton.Config({"BLOCK_C": 8, "BLOCK_S": 128}, num_warps=4),
        triton.Config({"BLOCK_C": 8, "BLOCK_S": 256}, num_warps=8),
        triton.Config({"BLOCK_C": 16, "BLOCK_S": 64}, num_warps=4),
        triton.Config({"BLOCK_C": 16, "BLOCK_S": 128}, num_warps=8),
    ],
    key=["C", "S"],
)
@triton.jit
def _cat_bn_silu_mean_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    rm_ptr,
    rv_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    mean_ptr,
    C0,
    C1,
    C2,
    C3,
    C,
    S,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_cb = tl.program_id(1)

    c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < C

    rm = tl.load(rm_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    rv = tl.load(rv_ptr + c, mask=c_mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    scale = weight * tl.rsqrt(rv + 1e-5)
    shift = bias - rm * scale

    sum_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    base_out = pid_n * C * S
    base0 = pid_n * C0 * S
    base1 = pid_n * C1 * S
    base2 = pid_n * C2 * S
    base3 = pid_n * C3 * S

    for s_block in range(0, tl.cdiv(S, BLOCK_S)):
        s = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s < S
        mask_cs = c_mask[:, None] & s_mask[None, :]

        vals = tl.zeros((BLOCK_C, BLOCK_S), dtype=tl.float32)

        mask0_c = c < C0
        offs0 = base0 + c[:, None] * S + s[None, :]
        vals += tl.load(x0_ptr + offs0, mask=mask0_c[:, None] & s_mask[None, :], other=0.0).to(tl.float32)

        c1_local = c - C0
        mask1_c = (c >= C0) & (c < C0 + C1)
        offs1 = base1 + c1_local[:, None] * S + s[None, :]
        vals += tl.load(x1_ptr + offs1, mask=mask1_c[:, None] & s_mask[None, :], other=0.0).to(tl.float32)

        c2_local = c - (C0 + C1)
        mask2_c = (c >= C0 + C1) & (c < C0 + C1 + C2)
        offs2 = base2 + c2_local[:, None] * S + s[None, :]
        vals += tl.load(x2_ptr + offs2, mask=mask2_c[:, None] & s_mask[None, :], other=0.0).to(tl.float32)

        c3_local = c - (C0 + C1 + C2)
        mask3_c = (c >= C0 + C1 + C2) & (c < C)
        offs3 = base3 + c3_local[:, None] * S + s[None, :]
        vals += tl.load(x3_ptr + offs3, mask=mask3_c[:, None] & s_mask[None, :], other=0.0).to(tl.float32)

        y = vals * scale[:, None] + shift[:, None]
        out = y * tl.sigmoid(y)
        out = tl.where(mask_cs, out, 0.0)

        offs_out = base_out + c[:, None] * S + s[None, :]
        tl.store(out_ptr + offs_out, out, mask=mask_cs)
        sum_acc += tl.sum(out, axis=1)

    mean = sum_acc / S
    tl.store(mean_ptr + pid_n * C + c, mean, mask=c_mask)


@torch.fx.wrap
def fused_cat_bn_silu_mean(running_mean, running_var, bias, weight, x0, x1, x2, x3):
    n = x0.shape[0]
    c0 = x0.shape[1]
    c1 = x1.shape[1]
    c2 = x2.shape[1]
    c3 = x3.shape[1]
    h = x0.shape[2]
    w = x0.shape[3]
    c = c0 + c1 + c2 + c3
    s = h * w

    out = torch.empty((n, c, h, w), device=x0.device, dtype=x0.dtype)
    mean = torch.empty((n, c, 1, 1), device=x0.device, dtype=x0.dtype)

    grid = lambda meta: (n, triton.cdiv(c, meta["BLOCK_C"]))
    _cat_bn_silu_mean_kernel[grid](
        x0,
        x1,
        x2,
        x3,
        running_mean,
        running_var,
        bias,
        weight,
        out,
        mean,
        c0,
        c1,
        c2,
        c3,
        c,
        s,
    )
    return out, mean


def replacement_func():
    return fused_cat_bn_silu_mean