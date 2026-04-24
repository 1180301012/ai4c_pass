import torch
import triton
import triton.language as tl


# ── Kernel 1: fused 7-input addition ────────────────────────────────────────

@triton.jit
def _add7_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, f_ptr, g_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(d_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    e = tl.load(e_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    f = tl.load(f_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + row_start + offsets, a + b + c + d + e + f + g, mask=mask)


def _run_add7(a, b, c, d, e, f, g):
    H = a.shape[-1]
    N_rows = a.numel() // H
    out = torch.empty_like(a)
    _add7_kernel[(N_rows,)](
        a, b, c, d, e, f, g, out,
        N=H,
        BLOCK_SIZE=256,
    )
    return out


# ── Kernel 2: fused embedding-sum + LayerNorm ───────────────────────────────
# H=768 = 3 × 256 → BLOCK_SIZE=256 divides evenly, no masking needed.
# Each program processes one (row, block) tile; grid = (N_rows * 3,).
# One full row needs all 3 programs so we use a single-pass with all 3 blocks.

@triton.jit
def _fused_sum_ln_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, f_ptr, g_ptr, h_ptr,
    w_ptr, bi_ptr, out_ptr,
    H,
    BLOCK_SIZE: tl.constexpr,  # = 256
):
    pid = tl.program_id(0)
    row_idx = pid // 3          # which row
    block_in_row = pid % 3      # 0, 1, or 2
    row_start = row_idx * H

    # Three consecutive 256-element chunks cover H=768 without masking
    o0 = tl.arange(0, BLOCK_SIZE)
    o1 = BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    o2 = 2 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load all 8 embedding tensors for this (row, block_in_row) tile
    a0 = tl.load(a_ptr + row_start + o0).to(tl.float32)
    a1 = tl.load(a_ptr + row_start + o1).to(tl.float32)
    a2 = tl.load(a_ptr + row_start + o2).to(tl.float32)
    b0 = tl.load(b_ptr + row_start + o0).to(tl.float32)
    b1 = tl.load(b_ptr + row_start + o1).to(tl.float32)
    b2 = tl.load(b_ptr + row_start + o2).to(tl.float32)
    c0 = tl.load(c_ptr + row_start + o0).to(tl.float32)
    c1 = tl.load(c_ptr + row_start + o1).to(tl.float32)
    c2 = tl.load(c_ptr + row_start + o2).to(tl.float32)
    d0 = tl.load(d_ptr + row_start + o0).to(tl.float32)
    d1 = tl.load(d_ptr + row_start + o1).to(tl.float32)
    d2 = tl.load(d_ptr + row_start + o2).to(tl.float32)
    e0 = tl.load(e_ptr + row_start + o0).to(tl.float32)
    e1 = tl.load(e_ptr + row_start + o1).to(tl.float32)
    e2 = tl.load(e_ptr + row_start + o2).to(tl.float32)
    f0 = tl.load(f_ptr + row_start + o0).to(tl.float32)
    f1 = tl.load(f_ptr + row_start + o1).to(tl.float32)
    f2 = tl.load(f_ptr + row_start + o2).to(tl.float32)
    g0 = tl.load(g_ptr + row_start + o0).to(tl.float32)
    g1 = tl.load(g_ptr + row_start + o1).to(tl.float32)
    g2 = tl.load(g_ptr + row_start + o2).to(tl.float32)
    h0 = tl.load(h_ptr + row_start + o0).to(tl.float32)
    h1 = tl.load(h_ptr + row_start + o1).to(tl.float32)
    h2 = tl.load(h_ptr + row_start + o2).to(tl.float32)

    # Sum of all 8 embeddings per chunk
    x0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0
    x1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1
    x2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2

    # LayerNorm: mean over H=768
    mean = (tl.sum(x0, 0) + tl.sum(x1, 0) + tl.sum(x2, 0)) / H

    # Variance
    var = (tl.sum((x0 - mean) * (x0 - mean), 0)
           + tl.sum((x1 - mean) * (x1 - mean), 0)
           + tl.sum((x2 - mean) * (x2 - mean), 0)) / H
    rstd = 1.0 / tl.sqrt(var + 1e-12)

    # Normalize + scale + shift and store per chunk
    # Load weight & bias (same for all chunks)
    w0 = tl.load(w_ptr + o0).to(tl.float32)
    w1 = tl.load(w_ptr + o1).to(tl.float32)
    w2 = tl.load(w_ptr + o2).to(tl.float32)
    bi0 = tl.load(bi_ptr + o0).to(tl.float32)
    bi1 = tl.load(bi_ptr + o1).to(tl.float32)
    bi2 = tl.load(bi_ptr + o2).to(tl.float32)

    tl.store(out_ptr + row_start + o0, (x0 - mean) * rstd * w0 + bi0)
    tl.store(out_ptr + row_start + o1, (x1 - mean) * rstd * w1 + bi1)
    tl.store(out_ptr + row_start + o2, (x2 - mean) * rstd * w2 + bi2)


def _run_fused_sum_ln(a, b, c, d, e, f, g, h, weight, bias):
    H = a.shape[-1]
    N_rows = a.numel() // H
    out = torch.empty_like(a)
    _fused_sum_ln_kernel[(N_rows * 3,)](
        a, b, c, d, e, f, g, h, weight, bias, out,
        H=H,
        BLOCK_SIZE=256,
    )
    return out


# ── Shared dispatch wrapper (single @torch.fx.wrap for all passes) ──────────

@torch.fx.wrap
def dispatch(a, b, c, d, e, f, g, h, i, j, route):
    if route == "full":
        # a=word_emb, b=pos_emb, c=emb_x1, d=emb_y1, e=emb_x2, f=emb_y2,
        # g=diff_y, h=diff_x, i=weight, j=bias
        return _run_fused_sum_ln(a, b, c, d, e, f, g, h, i, j)
    elif route == "add7_only":
        # fallback: sum 7 tensors only (add7 path)
        return _run_add7(a, b, c, d, e, f, g)
    elif route == "layernorm":
        # fallback: layernorm only (3 useful tensors: a=x, b=weight, c=bias)
        return _run_layernorm(a, b, c)


def _run_layernorm(x, weight, bias):
    H = x.shape[-1]
    N_rows = x.numel() // H
    out = torch.empty_like(x)
    _layernorm_kernel[(N_rows,)](
        x, out, weight, bias,
        N=H,
    )
    return out


# ── _layernorm_kernel used by the layernorm fallback route ──────────────────

@triton.jit
def _layernorm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / N
    diff = x - mean
    var = tl.sum(diff * diff, 0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-12)
    x_norm = diff * rstd

    w = tl.load(W_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w + b

    tl.store(Y_ptr + row_start + offsets, y, mask=mask)