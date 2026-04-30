import torch
import triton
import triton.language as tl


@triton.jit
def _wavlm_gru_rel_pos_fused_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_out,
    head_count,
    seq_len,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * 1 + tl.arange(0, 1)
    mask = offs < n_out

    hs = offs
    s = hs % seq_len
    h = hs // seq_len

    base_x = ((h * seq_len) + s) * 64
    k = tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + base_x[:, None] + k[None, :], mask=mask[:, None], other=0.0).to(tl.float32)

    idx0 = tl.arange(0, 4)
    idx1 = tl.arange(4, 8)

    w0 = tl.load(w_ptr + idx0[:, None] * 64 + k[None, :]).to(tl.float32)
    w1 = tl.load(w_ptr + idx1[:, None] * 64 + k[None, :]).to(tl.float32)
    b0 = tl.load(b_ptr + idx0).to(tl.float32)
    b1 = tl.load(b_ptr + idx1).to(tl.float32)

    acc0 = tl.sum(x[:, None, :] * w0[None, :, :], axis=2) + b0[None, :]
    acc1 = tl.sum(x[:, None, :] * w1[None, :, :], axis=2) + b1[None, :]

    s0 = tl.sum(acc0, axis=1)
    s1 = tl.sum(acc1, axis=1)

    sig0 = tl.sigmoid(s0)
    sig1 = tl.sigmoid(s1)

    c = tl.load(c_ptr + h, mask=mask, other=0.0).to(tl.float32)
    out = sig0 * (sig1 * c - 1.0) + 2.0

    out_offset = h * seq_len + s
    tl.store(out_ptr + out_offset, out.to(tl.float32), mask=mask)


@torch.fx.wrap
def wavlm_gru_rel_pos_fused(in_0, in_1, in_2, in_3, route):
    head_count = 12 if route == "h12" else 16
    seq_len = 199
    n_out = head_count * seq_len

    out = torch.empty((1, head_count, seq_len, 1), device=in_3.device, dtype=in_3.dtype)

    _wavlm_gru_rel_pos_fused_kernel[(n_out,)](
        in_3,
        in_1,
        in_0,
        in_2,
        out,
        n_out,
        head_count,
        seq_len,
        BLOCK_K=64,
    )
    return out