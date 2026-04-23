import torch
import triton
import triton.language as tl


# Match the full observable subgraph exactly.
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 32}, num_warps=4),
        triton.Config({"BLOCK_HW": 128, "BLOCK_C": 32}, num_warps=4),
        triton.Config({"BLOCK_HW": 256, "BLOCK_C": 32}, num_warps=8),
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_HW": 128, "BLOCK_C": 64}, num_warps=8),
    ],
    key=["HW", "C"],
)
@triton.jit
def _se_fused_kernel(
    x_ptr,
    se_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    K,
    HW,
    stride_x_n,
    stride_x_c,
    stride_x_h,
    stride_x_w,
    stride_se_n,
    stride_se_k,
    stride_w_c,
    stride_w_k,
    stride_b_c,
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    nc_per_batch = tl.cdiv(C, BLOCK_C)
    n_idx = pid_nc // nc_per_batch
    c_block = pid_nc % nc_per_batch
    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    c_mask = c_offsets < C
    hw_mask = hw_offsets < HW

    h_offsets = hw_offsets // W
    w_offsets_hw = hw_offsets - h_offsets * W

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    b_vals = tl.load(b_ptr + c_offsets * stride_b_c, mask=c_mask, other=0.0)
    acc += b_vals.to(tl.float32)

    k_offsets = tl.arange(0, 32)
    for k_base in range(0, 32, 32):
        k_idx = k_base + k_offsets
        k_mask = k_idx < K
        se_vals = tl.load(
            se_ptr + n_idx * stride_se_n + k_idx * stride_se_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        w_vals = tl.load(
            w_ptr + c_offsets[:, None] * stride_w_c + k_idx[None, :] * stride_w_k,
            mask=c_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w_vals * se_vals[None, :], axis=1)

    gate = tl.sigmoid(acc)

    x_vals = tl.load(
        x_ptr
        + n_idx * stride_x_n
        + c_offsets[:, None] * stride_x_c
        + h_offsets[None, :] * stride_x_h
        + w_offsets_hw[None, :] * stride_x_w,
        mask=c_mask[:, None] & hw_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    out_vals = x_vals * gate[:, None]
    out_vals = tl.maximum(out_vals, 0.0)
    out_vals = tl.minimum(out_vals, 6.0)

    tl.store(
        out_ptr
        + n_idx * stride_out_n
        + c_offsets[:, None] * stride_out_c
        + h_offsets[None, :] * stride_out_h
        + w_offsets_hw[None, :] * stride_out_w,
        out_vals,
        mask=c_mask[:, None] & hw_mask[None, :],
    )


@torch.fx.wrap
def _se_fused_impl(bias, weight, x, se):
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    K = se.shape[1]
    HW = H * W

    out = torch.empty_like(x)

    stride_se_n = se.stride(0)
    stride_se_k = se.stride(1)
    stride_w_c = weight.stride(0)
    stride_w_k = weight.stride(1)

    grid = lambda meta: (
        triton.cdiv(HW, meta["BLOCK_HW"]),
        N * triton.cdiv(C, meta["BLOCK_C"]),
    )

    _se_fused_kernel[grid](
        x,
        se,
        weight,
        bias,
        out,
        N,
        C,
        H,
        W,
        K,
        HW,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        stride_se_n,
        stride_se_k,
        stride_w_c,
        stride_w_k,
        bias.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def replacement_func():
    return _se_fused_impl