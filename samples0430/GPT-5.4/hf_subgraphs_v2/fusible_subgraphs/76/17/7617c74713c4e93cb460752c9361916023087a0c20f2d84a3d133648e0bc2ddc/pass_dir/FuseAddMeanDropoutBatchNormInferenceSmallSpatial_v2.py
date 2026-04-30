import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (tmp_8, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=1),
    ],
    key=["N", "C", "H", "W"],
)
@triton.jit
def fused_add_mean_bn_channel_block_kernel(
    running_mean_ptr,
    running_var_ptr,
    bias_ptr,
    weight_ptr,
    x_ptr,
    y_ptr,
    out_bn_ptr,
    out_mean_ptr,
    N,
    C,
    H,
    W,
    sxn,
    sxc,
    sxh,
    sxw,
    syn,
    syc,
    syh,
    syw,
    sobn0,
    sobn1,
    somean0,
    somean1,
    eps,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    c_block = tl.program_id(1)

    n = pid_nc
    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    hw = H * W
    hw_offsets = tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < hw
    h_idx = hw_offsets // W
    w_idx = hw_offsets % W

    x_base = n * sxn
    y_base = n * syn

    x_ptrs = x_ptr + x_base + c_offsets[:, None] * sxc + h_idx[None, :] * sxh + w_idx[None, :] * sxw
    y_ptrs = y_ptr + y_base + c_offsets[:, None] * syc + h_idx[None, :] * syh + w_idx[None, :] * syw
    elem_mask = c_mask[:, None] & hw_mask[None, :]

    x_vals = tl.load(x_ptrs, mask=elem_mask, other=0).to(tl.float32)
    y_vals = tl.load(y_ptrs, mask=elem_mask, other=0).to(tl.float32)
    mean_vals = tl.sum(x_vals + y_vals, axis=1) / hw

    running_mean = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0).to(tl.float32)
    running_var = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=1).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=1).to(tl.float32)

    bn_vals = (mean_vals - running_mean) * tl.rsqrt(running_var + eps) * weight + bias

    out_mean_ptrs = out_mean_ptr + n * somean0 + c_offsets * somean1
    out_bn_ptrs = out_bn_ptr + n * sobn0 + c_offsets * sobn1
    tl.store(out_mean_ptrs, mean_vals, mask=c_mask)
    tl.store(out_bn_ptrs, bn_vals, mask=c_mask)


@torch.fx.wrap
def fused_add_mean_bn_channel_block(running_mean, running_var, bias, weight, x, y):
    N, C, H, W = x.shape
    out_mean = torch.empty((N, C), device=x.device, dtype=x.dtype)
    out_bn = torch.empty((N, C), device=x.device, dtype=x.dtype)

    grid = (N, triton.cdiv(C, 16))
    fused_add_mean_bn_channel_block_kernel[grid](
        running_mean,
        running_var,
        bias,
        weight,
        x,
        y,
        out_bn,
        out_mean,
        N,
        C,
        H,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        out_bn.stride(0),
        out_bn.stride(1),
        out_mean.stride(0),
        out_mean.stride(1),
        1e-5,
    )
    return (out_bn, out_mean)


def replacement_func():
    return fused_add_mean_bn_channel_block