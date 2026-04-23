import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['spatial'],
)
@triton.jit
def _plane_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    feat_ptr,
    out_ptr,
    spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_plane = tl.program_id(1)

    b = pid_plane // 64
    c = pid_plane % 64

    k_offsets = tl.arange(0, 8)
    x = tl.load(input_ptr + b * 8 + k_offsets).to(tl.float32)
    w = tl.load(weight_ptr + c * 8 + k_offsets).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    gate = tl.sigmoid(tl.sum(x * w, axis=0) + bias)

    s_offsets = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = s_offsets < spatial
    base = pid_plane * spatial
    feat = tl.load(feat_ptr + base + s_offsets, mask=mask, other=0.0)
    out = feat * gate
    tl.store(out_ptr + base + s_offsets, out, mask=mask)


@triton.jit
def _batch1_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    feat_ptr,
    out_ptr,
    spatial,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_s = tl.program_id(0)

    k_offsets = tl.arange(0, 8)
    x = tl.load(input_ptr + k_offsets).to(tl.float32)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offsets < spatial

    for c_start in tl.static_range(0, 64, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < 64

        w = tl.load(
            weight_ptr + c_offsets[:, None] * 8 + k_offsets[None, :],
            mask=c_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        gate = tl.sigmoid(tl.sum(w * x[None, :], axis=1) + bias)

        mask = c_mask[:, None] & s_mask[None, :]
        ptrs = feat_ptr + c_offsets[:, None] * spatial + s_offsets[None, :]
        feat = tl.load(ptrs, mask=mask, other=0.0)
        out = feat * gate[:, None]
        tl.store(out_ptr + c_offsets[:, None] * spatial + s_offsets[None, :], out, mask=mask)


@torch.fx.wrap
def fused_linear_sigmoid_view_mul(in_0, in_1, in_2, in_3):
    batch = in_2.shape[0]
    h = in_3.shape[2]
    w = in_3.shape[3]
    spatial = h * w

    out = torch.empty_like(in_3)
    if batch == 1:
        block_s = 512
        grid = (triton.cdiv(spatial, block_s),)
        _batch1_kernel[grid](
            bias_ptr=in_0,
            weight_ptr=in_1,
            input_ptr=in_2,
            feat_ptr=in_3,
            out_ptr=out,
            spatial=spatial,
            BLOCK_S=block_s,
            BLOCK_C=4,
            num_warps=4,
        )
    else:
        grid = (triton.cdiv(spatial, 1024), batch * 64)
        _plane_kernel[grid](
            bias_ptr=in_0,
            weight_ptr=in_1,
            input_ptr=in_2,
            feat_ptr=in_3,
            out_ptr=out,
            spatial=spatial,
        )
    return out


def replacement_func():
    return fused_linear_sigmoid_view_mul