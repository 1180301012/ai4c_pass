import torch
import triton
import triton.language as tl


def pattern(in_5, in_6, in_2, in_1):
    """
    Match: add + layer_norm  (the only two ops that produce the model's
    observable output tmp_6).  slice + linear + tanh + view are left in the
    compiled graph; their FX-interpreter dispatch is already fast (~0.268 ms
    compiled baseline vs 0.190 ms eager = 0.71x, which is good).
    """
    tmp_5 = in_6 + in_5
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_2, in_1, 1e-12)
    return tmp_6


def replacement_args(in_5, in_6, in_2, in_1):
    return (in_5, in_6, in_2, in_1)


@triton.jit
def fused_add_layernorm_kernel(
    x_ptr, y_ptr, out_ptr, weight_ptr, bias_ptr,
    N:     tl.constexpr,
    eps:   tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row      = tl.program_id(0)
    row_base = row * N

    # ── Pass 1: elements [0, BLOCK_SIZE) ─────────────────────────────────────
    off_a = tl.arange(0, BLOCK_SIZE)
    x_a   = tl.load(x_ptr + row_base + off_a).to(tl.float32)
    y_a   = tl.load(y_ptr + row_base + off_a).to(tl.float32)
    z_a   = x_a + y_a
    s_z   = tl.sum(z_a, 0)
    s_z2  = tl.sum(z_a * z_a, 0)

    # ── Pass 2: elements [BLOCK_SIZE, 2*BLOCK_SIZE) ──────────────────────────
    off_b = BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_b   = tl.load(x_ptr + row_base + off_b).to(tl.float32)
    y_b   = tl.load(y_ptr + row_base + off_b).to(tl.float32)
    z_b   = x_b + y_b
    s_z   = s_z  + tl.sum(z_b, 0)
    s_z2  = s_z2 + tl.sum(z_b * z_b, 0)

    # ── Pass 3: elements [2*BLOCK_SIZE, N)  (3*128=384= N, exact) ────────────
    off_c = 2 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_c   = tl.load(x_ptr + row_base + off_c).to(tl.float32)
    y_c   = tl.load(y_ptr + row_base + off_c).to(tl.float32)
    z_c   = x_c + y_c
    s_z   = s_z  + tl.sum(z_c, 0)
    s_z2  = s_z2 + tl.sum(z_c * z_c, 0)

    # ── Normalisation ──────────────────────────────────────────────────────────
    mean = s_z  / N
    var  = s_z2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # ── Affine + store (pass 1) ────────────────────────────────────────────────
    w_a = tl.load(weight_ptr + off_a).to(tl.float32)
    b_a = tl.load(bias_ptr   + off_a).to(tl.float32)
    tl.store(out_ptr + row_base + off_a,
             ((z_a - mean) * rstd * w_a + b_a).to(x_ptr.dtype.element_ty))

    # ── Affine + store (pass 2) ────────────────────────────────────────────────
    w_b = tl.load(weight_ptr + off_b).to(tl.float32)
    b_b = tl.load(bias_ptr   + off_b).to(tl.float32)
    tl.store(out_ptr + row_base + off_b,
             ((z_b - mean) * rstd * w_b + b_b).to(x_ptr.dtype.element_ty))

    # ── Affine + store (pass 3) ────────────────────────────────────────────────
    w_c = tl.load(weight_ptr + off_c).to(tl.float32)
    b_c = tl.load(bias_ptr   + off_c).to(tl.float32)
    tl.store(out_ptr + row_base + off_c,
             ((z_c - mean) * rstd * w_c + b_c).to(x_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_add_layernorm(in_5, in_6, in_2, in_1):
    """
    Replacement for add + layer_norm.
    3-pass kernel: BLOCK_SIZE=128, N=384=3×128 (zero masking waste).
    N and eps are constexpr to reduce runtime arg count.
    num_warps=1: pure warp-shuffle reduction; no shared-memory sync.
    """
    num_rows = in_6.numel() // 384   # = 578
    out      = torch.empty_like(in_6)

    fused_add_layernorm_kernel[(num_rows,)](
        in_6, in_5, out, in_2, in_1,
        N=384, eps=1e-12,
        BLOCK_SIZE=128,
        num_warps=1,
        num_stages=2,
    )
    return out


def replacement_func():
    return fused_add_layernorm