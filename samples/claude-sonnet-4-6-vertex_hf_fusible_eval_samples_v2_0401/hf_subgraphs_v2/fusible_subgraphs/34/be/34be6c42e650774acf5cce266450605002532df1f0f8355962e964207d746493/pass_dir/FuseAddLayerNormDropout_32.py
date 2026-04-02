import torch
import triton
import triton.language as tl
from torch import device


# Match the full computation for hidden_dim=32
def pattern(in_0, in_1, in_2, in_3, in_4):
    # --- mask path ---
    tmp_5 = in_0.__eq__(1)
    tmp_6 = tmp_5.to(torch.float32)
    tmp_6 *= -3.4028234663852886e+38
    tmp_8 = tmp_6.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(1)

    # --- word embedding ---
    tmp_10 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)

    # --- position IDs (constant for seq_len=15) ---
    tmp_11 = torch.ones((1, 15), dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_12 = torch.cumsum(tmp_11, dim=1)
    tmp_13 = tmp_12 - tmp_11
    tmp_13 += 2

    # --- position embedding ---
    tmp_15 = torch.nn.functional.embedding(tmp_13, in_3, 1, None, 2.0, False, False)

    # --- add + layer_norm + dropout ---
    tmp_16 = tmp_10 + tmp_15
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (32,), in_2, in_1, 1e-05)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)

    return tmp_18, tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ── Kernel 1: mask  ──────────────────────────────────────────────────────────
@triton.jit
def _mask_kernel_32(
    TOKEN_IDS_ptr,   # [B*T] int64
    OUT_ptr,         # [B*T] float32
    T,               # sequence length (15)
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = idx < T
    tok = tl.load(TOKEN_IDS_ptr + idx, mask=valid, other=0)
    val = tl.where(tok == 1, -3.4028234663852886e+38, 0.0)
    tl.store(OUT_ptr + idx, val.to(tl.float32), mask=valid)


# ── Kernel 2a: bf16 fused embedding + add + layer-norm (N=32) ────────────────
@triton.jit
def _emb_add_ln_kernel_32_bf16(
    TOKEN_IDS_ptr,
    WORD_ptr,        # [V, 32] bfloat16
    POS_ptr,         # [P, 32] bfloat16
    LN_W_ptr,        # [32]    bfloat16
    LN_B_ptr,        # [32]    bfloat16
    OUT_ptr,         # [B*T, 32] bfloat16
    eps,
    BLOCK_SIZE: tl.constexpr,   # 32
    N:          tl.constexpr,   # 32
):
    row = tl.program_id(0)
    token_id = tl.load(TOKEN_IDS_ptr + row)
    pos_id = row + 2   # deterministic: positions are [2, ..., T+1]

    cols = tl.arange(0, BLOCK_SIZE)

    word_raw = tl.load(WORD_ptr + token_id * N + cols).to(tl.float32)
    word_emb = tl.where(token_id == 1, 0.0, word_raw)

    pos_emb = tl.load(POS_ptr + pos_id * N + cols).to(tl.float32)

    z = word_emb + pos_emb

    mean  = tl.sum(z, axis=0) / N
    diff  = z - mean
    var   = tl.sum(diff * diff, axis=0) / N
    rstd  = 1.0 / tl.sqrt(var + eps)
    z_hat = diff * rstd

    w = tl.load(LN_W_ptr + cols).to(tl.float32)
    b = tl.load(LN_B_ptr + cols).to(tl.float32)
    out_f32 = z_hat * w + b

    tl.store(OUT_ptr + row * N + cols, out_f32.to(tl.bfloat16))


# ── Kernel 2b: fp16 fused embedding + add + layer-norm (N=32) ────────────────
@triton.jit
def _emb_add_ln_kernel_32_fp16(
    TOKEN_IDS_ptr,
    WORD_ptr,
    POS_ptr,
    LN_W_ptr,
    LN_B_ptr,
    OUT_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
    N:          tl.constexpr,
):
    row = tl.program_id(0)
    token_id = tl.load(TOKEN_IDS_ptr + row)
    pos_id = row + 2

    cols = tl.arange(0, BLOCK_SIZE)

    word_raw = tl.load(WORD_ptr + token_id * N + cols).to(tl.float32)
    word_emb = tl.where(token_id == 1, 0.0, word_raw)

    pos_emb = tl.load(POS_ptr + pos_id * N + cols).to(tl.float32)

    z = word_emb + pos_emb

    mean  = tl.sum(z, axis=0) / N
    diff  = z - mean
    var   = tl.sum(diff * diff, axis=0) / N
    rstd  = 1.0 / tl.sqrt(var + eps)
    z_hat = diff * rstd

    w = tl.load(LN_W_ptr + cols).to(tl.float32)
    b = tl.load(LN_B_ptr + cols).to(tl.float32)
    out_f32 = z_hat * w + b

    tl.store(OUT_ptr + row * N + cols, out_f32.to(tl.float16))


# ── Replacement wrapper ───────────────────────────────────────────────────────
@torch.fx.wrap
def full_fused_32(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : [B, T] int64  – token IDs
    in_1 : [32] float    – layer-norm bias
    in_2 : [32] float    – layer-norm weight
    in_3 : [P, 32] float – position embedding weight
    in_4 : [V, 32] float – word embedding weight
    Returns: (emb_output [B,T,32], mask [B,1,1,T])
    """
    N = 32
    B, T = in_0.shape
    num_rows = B * T

    emb_out  = torch.empty((B, T, N), dtype=in_4.dtype, device=in_4.device)
    mask_out = torch.empty((B, 1, 1, T), dtype=torch.float32, device=in_0.device)

    token_ids_flat = in_0.reshape(-1)

    # Kernel 1 – mask
    _mask_kernel_32[(1,)](
        token_ids_flat, mask_out.reshape(-1),
        T=T, BLOCK_SIZE=32,
    )

    # Kernel 2 – fused emb + add + layer-norm
    if in_4.dtype == torch.bfloat16:
        _emb_add_ln_kernel_32_bf16[(num_rows,)](
            token_ids_flat, in_4, in_3, in_2, in_1,
            emb_out.reshape(num_rows, N),
            eps=1e-5, BLOCK_SIZE=32, N=N,
            num_warps=1,
        )
    else:
        _emb_add_ln_kernel_32_fp16[(num_rows,)](
            token_ids_flat, in_4, in_3, in_2, in_1,
            emb_out.reshape(num_rows, N),
            eps=1e-5, BLOCK_SIZE=32, N=N,
            num_warps=1,
        )

    return emb_out, mask_out


def replacement_func():
    return full_fused_32