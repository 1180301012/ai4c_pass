import torch
import triton
import triton.language as tl


# Match only the pointwise tail, leaving the tiny 1x1 conv to the backend.
def pattern(in_2, conv2d_out):
    tmp_3 = conv2d_out.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(in_2, conv2d_out):
    return (in_2, conv2d_out)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 32}, num_warps=2),
        triton.Config({"BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_C": 128}, num_warps=4),
    ],
    key=["C", "HW"],
)
@triton.jit
def _sigmoid_mul_hardtanh_kernel(
    x_ptr,
    gate_pre_ptr,
    out_ptr,
    C,
    W,
    HW,
    stride_x_n,
    stride_x_c,
    stride_x_h,
    stride_x_w,
    stride_gate_n,
    stride_gate_c,
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    NUM_HW_BLOCKS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_nc = tl.program_id(0)

    nc_per_batch = tl.cdiv(C, BLOCK_C)
    n_idx = pid_nc // nc_per_batch
    c_block = pid_nc % nc_per_batch

    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    gate = tl.load(
        gate_pre_ptr + n_idx * stride_gate_n + c_offsets * stride_gate_c,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.sigmoid(gate)

    for hw_block_idx in tl.static_range(0, NUM_HW_BLOCKS):
        hw_offsets = hw_block_idx * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        h_offsets = hw_offsets // W
        w_offsets = hw_offsets - h_offsets * W

        x_vals = tl.load(
            x_ptr
            + n_idx * stride_x_n
            + c_offsets[:, None] * stride_x_c
            + h_offsets[None, :] * stride_x_h
            + w_offsets[None, :] * stride_x_w,
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
            + w_offsets[None, :] * stride_out_w,
            out_vals,
            mask=c_mask[:, None] & hw_mask[None, :],
        )


@torch.fx.wrap
def _sigmoid_mul_hardtanh_impl(x, gate_pre):
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    HW = H * W

    out = torch.empty_like(x)

    grid = lambda meta: (N * triton.cdiv(C, meta["BLOCK_C"]),)

    _sigmoid_mul_hardtanh_kernel[grid](
        x,
        gate_pre,
        out,
        C,
        W,
        HW,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        gate_pre.stride(0),
        gate_pre.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        NUM_HW_BLOCKS=(HW + 255) // 256,
        BLOCK_HW=256,
    )
    return out


def replacement_func():
    return _sigmoid_mul_hardtanh_impl