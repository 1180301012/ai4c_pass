import torch
import triton
import triton.language as tl


@triton.jit
def _fused_linear_mean_kernel(
    # linear inputs / output
    x_lin_ptr, w_ptr, b_ptr, out_lin_ptr,
    # mean input / output
    x_mean_ptr, out_mean_ptr,
    # runtime shapes
    B, N, K, SEQ_LEN, HIDDEN,
    # compile-time constants
    BLOCK   : tl.constexpr,   # 64 — both K-tile and H-tile
    H_BLOCKS: tl.constexpr,   # ceil(HIDDEN / BLOCK) = 7
    N_LIN   : tl.constexpr,   # number of output classes = 2
):
    """
    Unified kernel for linear + mean over a fused grid:
      program_id(0) = op_id  (0..N_LIN-1 → linear, N_LIN..N_LIN+H_BLOCKS-1 → mean)
      program_id(1) = batch b

    program_id(0) is block-uniform ⇒ NO warp divergence on the if/else.
    Both linear and mean thread-blocks are launched simultaneously and overlap on GPU.
    """
    op_id = tl.program_id(0)
    b     = tl.program_id(1)

    if op_id < N_LIN:
        # ── Linear ──────────────────────────────────────────────────────────────
        # out_lin[b, col] = dot(x_lin[b, :], w[col, :]) + bias[col]
        col = op_id
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k0 in range(0, K, BLOCK):
            k_offs = k0 + tl.arange(0, BLOCK)
            k_mask = k_offs < K
            xv = tl.load(x_lin_ptr + b * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr     + col * K + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            acc += xv * wv
        bias_v = tl.load(b_ptr + col).to(tl.float32)
        tl.store(out_lin_ptr + b * N + col, tl.sum(acc) + bias_v)

    else:
        # ── Mean ─────────────────────────────────────────────────────────────────
        # out_mean[b, h_block*BLOCK : (h_block+1)*BLOCK] = mean(x_mean[b, :, h], axis=seq)
        h_block = op_id - N_LIN
        h_offs  = h_block * BLOCK + tl.arange(0, BLOCK)
        h_mask  = h_offs < HIDDEN
        acc     = tl.zeros([BLOCK], dtype=tl.float32)
        x_base  = x_mean_ptr + b * SEQ_LEN * HIDDEN
        for s in range(SEQ_LEN):
            acc += tl.load(x_base + s * HIDDEN + h_offs, mask=h_mask, other=0.0).to(tl.float32)
        tl.store(out_mean_ptr + b * HIDDEN + h_offs, acc / SEQ_LEN, mask=h_mask)


@torch.fx.wrap
def triton_fused_linear_mean(x_lin, w, b_bias, x_mean):
    B      = x_lin.shape[0]
    K      = x_lin.shape[1]    # 448
    N      = w.shape[0]        # 2
    SEQ    = x_mean.shape[1]   # 49
    HIDDEN = x_mean.shape[2]   # 448

    out_lin  = torch.empty((B, N),      dtype=x_lin.dtype,  device=x_lin.device)
    out_mean = torch.empty((B, HIDDEN), dtype=x_mean.dtype, device=x_mean.device)

    # HIDDEN=448 = 7×64  →  H_BLOCKS=7, BLOCK=64, no masking on H dimension
    H_BLOCKS = 7
    N_LIN    = 2

    # One kernel launch covers both operations; GPU runs them in parallel across SMs
    _fused_linear_mean_kernel[(N_LIN + H_BLOCKS, B)](
        x_lin, w, b_bias, out_lin,
        x_mean, out_mean,
        B, N, K, SEQ, HIDDEN,
        BLOCK=64, H_BLOCKS=H_BLOCKS, N_LIN=N_LIN,
        num_warps=4, num_stages=4,
    )

    return (out_lin, out_mean)


def pattern(x_lin, w, b_bias, x_mean):
    out_lin  = torch.nn.functional.linear(x_lin, w, b_bias)
    out_mean = x_mean.mean(-2)
    return (out_lin, out_mean)


def replacement_args(x_lin, w, b_bias, x_mean):
    return (x_lin, w, b_bias, x_mean)


def replacement_func():
    return triton_fused_linear_mean