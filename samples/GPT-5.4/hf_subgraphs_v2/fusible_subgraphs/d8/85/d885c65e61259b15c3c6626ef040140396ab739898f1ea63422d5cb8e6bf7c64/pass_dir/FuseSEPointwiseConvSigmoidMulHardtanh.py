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
        triton.Config({"BLOCK_C": 32}, num_warps=2),
        triton.Config({"BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_C": 128}, num_warps=4),
    ],
    key=["C", "HW"],
)
@triton.jit
def _se_persistent_kernel(
    x_ptr,
    se_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C,
    K,
    W,
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
    NUM_HW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    nc_per_batch = tl.cdiv(C, BLOCK_C)
    n_idx = pid // nc_per_batch
    c_block = pid - n_idx * nc_per_batch

    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.arange(0, BLOCK_K)
    c_mask = c_offsets < C
    k_mask = k_offsets < K

    acc = tl.load(b_ptr + c_offsets * stride_b_c, mask=c_mask, other=0.0).to(tl.float32)
    se_vals = tl.load(
        se_ptr + n_idx * stride_se_n + k_offsets * stride_se_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    w_vals = tl.load(
        w_ptr + c_offsets[:, None] * stride_w_c + k_offsets[None, :] * stride_w_k,
        mask=c_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc += tl.sum(w_vals * se_vals[None, :], axis=1)
    gate = tl.sigmoid(acc)

    for hw_block_idx in tl.static_range(0, NUM_HW_BLOCKS):
        hw_offsets = hw_block_idx * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        h_offsets = hw_offsets // W
        w_offsets_hw = hw_offsets - h_offsets * W

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
    # x: [N, C, H, W]
    # se: [N, K, 1, 1]
    # weight: [C, K, 1, 1]
    # bias: [C]
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    K = se.shape[1]
    HW = H * W

    out = torch.empty_like(x)

    grid = lambda meta: (N * triton.cdiv(C, meta["BLOCK_C"]),)
    _se_persistent_kernel[grid](
        x,
        se,
        weight,
        bias,
        out,
        C,
        K,
        W,
        HW,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        se.stride(0),
        se.stride(1),
        weight.stride(0),
        weight.stride(1),
        bias.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        NUM_HW_BLOCKS=(HW + 127) // 128,
        BLOCK_K=32,
        BLOCK_HW=128,
    )
    return out


def replacement_func():
    return _se_fused_impl