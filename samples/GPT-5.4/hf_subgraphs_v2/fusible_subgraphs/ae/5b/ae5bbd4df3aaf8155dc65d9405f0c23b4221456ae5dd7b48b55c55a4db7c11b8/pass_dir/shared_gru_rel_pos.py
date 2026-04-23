import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 256}, num_warps=8, num_stages=2),
    ],
    key=["M"],
)
@triton.jit
def fused_gru_rel_pos_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    H,
    L,
    stride_x0,
    stride_x1,
    stride_x2,
    stride_x3,
    stride_w0,
    stride_w1,
    stride_b0,
    stride_c0,
    stride_c1,
    stride_c2,
    stride_c3,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    M,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < M

    h_idx = rows // L
    l_idx = rows - h_idx * L

    k = tl.arange(0, 64)
    x_offsets = (
        0 * stride_x0
        + h_idx[:, None] * stride_x1
        + l_idx[:, None] * stride_x2
        + k[None, :] * stride_x3
    )
    x = tl.load(x_ptr + x_offsets, mask=mask[:, None], other=0.0).to(tl.float32)

    w0 = tl.load(w_ptr + 0 * stride_w0 + k * stride_w1).to(tl.float32)
    w1 = tl.load(w_ptr + 1 * stride_w0 + k * stride_w1).to(tl.float32)
    w2 = tl.load(w_ptr + 2 * stride_w0 + k * stride_w1).to(tl.float32)
    w3 = tl.load(w_ptr + 3 * stride_w0 + k * stride_w1).to(tl.float32)
    w4 = tl.load(w_ptr + 4 * stride_w0 + k * stride_w1).to(tl.float32)
    w5 = tl.load(w_ptr + 5 * stride_w0 + k * stride_w1).to(tl.float32)
    w6 = tl.load(w_ptr + 6 * stride_w0 + k * stride_w1).to(tl.float32)
    w7 = tl.load(w_ptr + 7 * stride_w0 + k * stride_w1).to(tl.float32)

    wg0 = w0 + w1 + w2 + w3
    wg1 = w4 + w5 + w6 + w7
    bias0 = (
        tl.load(b_ptr + 0 * stride_b0).to(tl.float32)
        + tl.load(b_ptr + 1 * stride_b0).to(tl.float32)
        + tl.load(b_ptr + 2 * stride_b0).to(tl.float32)
        + tl.load(b_ptr + 3 * stride_b0).to(tl.float32)
    )
    bias1 = (
        tl.load(b_ptr + 4 * stride_b0).to(tl.float32)
        + tl.load(b_ptr + 5 * stride_b0).to(tl.float32)
        + tl.load(b_ptr + 6 * stride_b0).to(tl.float32)
        + tl.load(b_ptr + 7 * stride_b0).to(tl.float32)
    )

    acc0 = tl.sum(x * wg0[None, :], axis=1) + bias0
    acc1 = tl.sum(x * wg1[None, :], axis=1) + bias1
    sig0 = tl.sigmoid(acc0)
    sig1 = tl.sigmoid(acc1)

    c_offsets = 0 * stride_c0 + h_idx * stride_c1 + 0 * stride_c2 + 0 * stride_c3
    c = tl.load(c_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
    out = sig0 * (sig1 * c - 1.0) + 2.0

    o_offsets = 0 * stride_o0 + h_idx * stride_o1 + l_idx * stride_o2 + 0 * stride_o3
    tl.store(out_ptr + o_offsets, out, mask=mask)


@torch.fx.wrap
def fused_wavlm_gru_rel_pos(in_0, in_1, in_2, in_3):
    H = in_3.shape[1]
    L = in_3.shape[2]
    M = H * L
    out = torch.empty((1, H, L, 1), device=in_3.device, dtype=in_3.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    fused_gru_rel_pos_kernel[grid](
        in_3,
        in_1,
        in_0,
        in_2,
        out,
        H,
        L,
        in_3.stride(0),
        in_3.stride(1),
        in_3.stride(2),
        in_3.stride(3),
        in_1.stride(0),
        in_1.stride(1),
        in_0.stride(0),
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        M,
    )
    return out