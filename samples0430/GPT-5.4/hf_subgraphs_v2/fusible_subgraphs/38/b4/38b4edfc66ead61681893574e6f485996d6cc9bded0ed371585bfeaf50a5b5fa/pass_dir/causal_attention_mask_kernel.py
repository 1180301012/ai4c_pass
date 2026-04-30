import torch
import triton
import triton.language as tl

NEG_INF = -3.4028234663852886e+38


@triton.jit
def _causal_attention_mask_kernel(
    attn_mask_ptr,
    out_ptr,
    N,
    stride_attn_b,
    stride_attn_n,
    stride_out_b,
    stride_out_h,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < N
    n_mask = offs_n < N
    mask_2d = m_mask[:, None] & n_mask[None, :]

    causal_keep = offs_n[None, :] < (offs_m[:, None] + 1)

    attn = tl.load(attn_mask_ptr + offs_n * stride_attn_n, mask=n_mask, other=0)
    pad_mask = attn != 1

    out = tl.full((BLOCK_M, BLOCK_N), NEG_INF, tl.float32)
    out = tl.where(causal_keep, 0.0, out)
    out = tl.where(pad_mask[None, :], NEG_INF, out)

    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, out, mask=mask_2d)


@torch.fx.wrap
def causal_attention_mask_dispatch(in_0, route):
    # route is expected to be one of: "seq9", "seq13"
    n = in_0.shape[-1]
    out = torch.empty((1, 1, n, n), device=in_0.device, dtype=torch.float32)

    # Specialized launch parameters for tiny masks.
    if route == "seq9":
        block_m = 16
        block_n = 16
    elif route == "seq13":
        block_m = 16
        block_n = 16
    else:
        block_m = 16
        block_n = 16

    grid = (triton.cdiv(n, block_m) * triton.cdiv(n, block_n),)
    _causal_attention_mask_kernel[grid](
        in_0,
        out,
        n,
        in_0.stride(0),
        in_0.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out