import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_QY": 4, "BLOCK_KX": 2, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_QY": 4, "BLOCK_KX": 4, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QY": 8, "BLOCK_KX": 2, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_QY": 8, "BLOCK_KX": 4, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QY": 8, "BLOCK_KX": 8, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_QY": 8, "BLOCK_KX": 4, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _botnet_rel_logits_kernel(
    q_ptr,
    rel_ptr,
    bias_ptr,
    out_ptr,
    stride_q_b,
    stride_q_y,
    stride_q_x,
    stride_q_d,
    stride_bias_b,
    stride_bias_qx,
    stride_bias_qy,
    stride_bias_kx,
    stride_bias_ky,
    S,
    N: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    BLOCK_QY: tl.constexpr,
    BLOCK_KX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_qy = tl.program_id(0)
    pid_kx = tl.program_id(1)
    pid_z = tl.program_id(2)
    pid_b = pid_z // N
    qx = pid_z % N

    qy = pid_qy * BLOCK_QY + tl.arange(0, BLOCK_QY)
    kx = pid_kx * BLOCK_KX + tl.arange(0, BLOCK_KX)
    ky = tl.arange(0, N)

    qy_mask = qy < N
    kx_mask = kx < N
    rel_cols = kx + (N - 1 - qx)

    acc = tl.zeros((BLOCK_QY, BLOCK_KX), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        d_mask = d < D

        q_vals = tl.load(
            q_ptr + pid_b * stride_q_b + qy[:, None] * stride_q_y + qx * stride_q_x + d[None, :] * stride_q_d,
            mask=qy_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        rel_vals = tl.load(
            rel_ptr + d[:, None] * R + rel_cols[None, :],
            mask=d_mask[:, None] & kx_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(q_vals, rel_vals)

    bias_vals = tl.load(
        bias_ptr + pid_b * stride_bias_b + qx * stride_bias_qx + qy[:, None, None] * stride_bias_qy + kx[None, :, None] * stride_bias_kx + ky[None, None, :] * stride_bias_ky,
        mask=qy_mask[:, None, None] & kx_mask[None, :, None],
        other=0.0,
    ).to(tl.float32)

    q_flat = qx * N + qy
    out_offsets = (((pid_b * S) + q_flat[:, None, None]) * S + kx[None, :, None] * N + ky[None, None, :])
    out_vals = acc[:, :, None] + bias_vals
    tl.store(out_ptr + out_offsets, out_vals, mask=qy_mask[:, None, None] & kx_mask[None, :, None])


def _botnet_rel_logits_impl(in_1, in_2, in_3, n):
    b = in_1.shape[0]
    s = n * n
    out = torch.empty((b, s, s), device=in_2.device, dtype=in_2.dtype)

    grid = lambda META: (triton.cdiv(n, META["BLOCK_QY"]), triton.cdiv(n, META["BLOCK_KX"]), b * n)
    _botnet_rel_logits_kernel[grid](
        q_ptr=in_1,
        rel_ptr=in_3,
        bias_ptr=in_2,
        out_ptr=out,
        stride_q_b=in_1.stride(0),
        stride_q_y=in_1.stride(1),
        stride_q_x=in_1.stride(2),
        stride_q_d=in_1.stride(3),
        stride_bias_b=in_2.stride(0),
        stride_bias_qx=in_2.stride(1),
        stride_bias_qy=in_2.stride(2),
        stride_bias_kx=in_2.stride(3),
        stride_bias_ky=in_2.stride(4),
        S=s,
        N=n,
        R=2 * n - 1,
        D=in_1.shape[-1],
    )
    return out


@torch.fx.wrap
def botnet_rel_logits_dispatch(in_1, in_2, in_3, route):
    if route == "botnet_rel_logits_16":
        return _botnet_rel_logits_impl(in_1, in_2, in_3, 16)
    if route == "botnet_rel_logits_8":
        return _botnet_rel_logits_impl(in_1, in_2, in_3, 8)
    raise RuntimeError(f"Unsupported route: {route}")