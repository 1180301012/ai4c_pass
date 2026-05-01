import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Fused Embedding Lookup kernel  (word + token-type + position, then add)
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 16}),
        triton.Config({'num_warps': 8}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 2}),
    ],
    key=['N_ROWS', 'HIDDEN'],
)
@triton.jit
def _fused_embed_kernel(
    word_ids_ptr, tt_ids_ptr, pos_ids_ptr,
    word_emb_ptr, tt_emb_ptr, pos_emb_ptr,
    out_ptr,
    N_ROWS, seq_len,
    HIDDEN: tl.constexpr,
    pos_batch_stride,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    s = row % seq_len

    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN

    word_idx = tl.load(word_ids_ptr + b * seq_len + s)
    tt_idx   = tl.load(tt_ids_ptr   + b * seq_len + s)
    pos_idx  = tl.load(pos_ids_ptr  + b * pos_batch_stride + s)

    x = tl.load(word_emb_ptr + word_idx * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(word_idx == 0, tl.zeros([BLOCK], dtype=tl.float32), x)
    x = x + tl.load(tt_emb_ptr + tt_idx * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + tl.load(pos_emb_ptr + pos_idx * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)

    if IS_FP16:
        tl.store(out_ptr + row * HIDDEN + cols, x.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + row * HIDDEN + cols, x.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row * HIDDEN + cols, x, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Layer-Norm kernel
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 16}),
        triton.Config({'num_warps': 8}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 2}),
    ],
    key=['N_ROWS', 'HIDDEN'],
)
@triton.jit
def _layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N_ROWS,
    HIDDEN: tl.constexpr,
    eps,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN

    x = tl.load(x_ptr + row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
    x_valid = tl.where(mask, x, 0.0)
    mean     = tl.sum(x_valid, 0) / HIDDEN
    xc       = tl.where(mask, x - mean, 0.0)
    var      = tl.sum(xc * xc, 0) / HIDDEN
    rstd     = 1.0 / tl.sqrt(var + eps)
    x_norm   = tl.where(mask, xc * rstd, 0.0)

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(mask, x_norm * w + b, 0.0)

    if IS_FP16:
        tl.store(out_ptr + row * HIDDEN + cols, y.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + row * HIDDEN + cols, y.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row * HIDDEN + cols, y, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Python wrappers
# ──────────────────────────────────────────────────────────────────────────────

def _dtype_flags(dtype):
    return (dtype == torch.float16), (dtype == torch.bfloat16)


def _fused_embedding_forward(word_ids, tt_ids, pos_ids,
                              word_emb, tt_emb, pos_emb):
    B        = int(word_ids.shape[0])
    S        = int(word_ids.shape[1])
    H        = int(word_emb.shape[1])
    N_ROWS   = B * S
    dtype    = word_emb.dtype
    device   = word_emb.device
    pos_bs   = 0 if pos_ids.shape[0] == 1 else S
    is16, ib = _dtype_flags(dtype)
    BLOCK    = max(triton.next_power_of_2(H), 16)

    # Allocate directly as [B, S, H] – no reshape/view needed
    out = torch.empty((B, S, H), dtype=dtype, device=device)
    _fused_embed_kernel[(N_ROWS,)](
        word_ids, tt_ids, pos_ids,
        word_emb, tt_emb, pos_emb,
        out,
        N_ROWS, S, H, pos_bs, is16, ib, BLOCK,
    )
    return out  # [B, S, H] – contiguous, same memory layout as [N_ROWS, H]


def _layer_norm_forward(x, ln_w, ln_b, H):
    # Compute N_ROWS without reshape (safe for PoisonDispatchTensor)
    N_ROWS = 1
    for d in x.shape[:-1]:
        N_ROWS = N_ROWS * int(d)
    dtype    = x.dtype
    device   = x.device
    is16, ib = _dtype_flags(dtype)
    BLOCK    = max(triton.next_power_of_2(H), 16)

    # Output same shape as input – no reshape/view
    out = torch.empty(x.shape, dtype=dtype, device=device)
    _layer_norm_kernel[(N_ROWS,)](
        x, ln_w, ln_b, out,
        N_ROWS, H, 1e-12, is16, ib, BLOCK,
    )
    return out  # Same shape as x


# ──────────────────────────────────────────────────────────────────────────────
# Shared dispatch function  (ONE replacement_func for all passes)
# ──────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_embed_add_ln(*args):
    """Route-based dispatch. Last arg is the route string."""
    route = args[-1]
    if route == "emb":
        # args: word_emb, word_ids, tt_emb, tt_ids, pos_emb, pos_ids
        word_emb, word_ids, tt_emb, tt_ids, pos_emb, pos_ids = args[:6]
        return _fused_embedding_forward(word_ids, tt_ids, pos_ids,
                                        word_emb, tt_emb, pos_emb)
    elif route == "ln_1024":
        x, ln_w, ln_b = args[:3]
        return _layer_norm_forward(x, ln_w, ln_b, 1024)
    elif route == "ln_768":
        x, ln_w, ln_b = args[:3]
        return _layer_norm_forward(x, ln_w, ln_b, 768)
    elif route == "ln_64":
        x, ln_w, ln_b = args[:3]
        return _layer_norm_forward(x, ln_w, ln_b, 64)
    elif route == "ln_32":
        x, ln_w, ln_b = args[:3]
        return _layer_norm_forward(x, ln_w, ln_b, 32)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy helpers kept for backward compat (unused by new passes)
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 16}),
        triton.Config({'num_warps': 8}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 2}),
    ],
    key=['N_ROWS', 'HIDDEN'],
)
@triton.jit
def _embed_add_ln_2out_kernel(
    word_ids_ptr, tt_ids_ptr, pos_ids_ptr,
    word_emb_ptr, tt_emb_ptr, pos_emb_ptr,
    ln_w_ptr, ln_b_ptr,
    pre_ln_ptr, ln_out_ptr,
    N_ROWS, seq_len,
    HIDDEN: tl.constexpr,
    pos_batch_stride,
    eps,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    s = row % seq_len

    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN

    # Load indices
    word_idx = tl.load(word_ids_ptr + b * seq_len + s)
    tt_idx = tl.load(tt_ids_ptr + b * seq_len + s)
    pos_idx = tl.load(pos_ids_ptr + b * pos_batch_stride + s)

    # Load word embedding (with padding_idx=0 zeroing)
    x = tl.load(word_emb_ptr + word_idx * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(word_idx == 0, tl.zeros([BLOCK], dtype=tl.float32), x)

    # Add token type embedding
    x = x + tl.load(tt_emb_ptr + tt_idx * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)

    # Add position embedding
    x = x + tl.load(pos_emb_ptr + pos_idx * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)

    # Store pre-LN
    if IS_FP16:
        tl.store(pre_ln_ptr + row * HIDDEN + cols, x.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(pre_ln_ptr + row * HIDDEN + cols, x.to(tl.bfloat16), mask=mask)
    else:
        tl.store(pre_ln_ptr + row * HIDDEN + cols, x, mask=mask)

    # Layer norm: compute mean and variance over valid elements
    x_valid = tl.where(mask, x, 0.0)
    mean = tl.sum(x_valid, 0) / HIDDEN
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, 0) / HIDDEN
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = tl.where(mask, xc * rstd, 0.0)

    # Apply LN weight and bias
    w = tl.load(ln_w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b_v = tl.load(ln_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(mask, x_norm * w + b_v, 0.0)

    # Store LN output
    if IS_FP16:
        tl.store(ln_out_ptr + row * HIDDEN + cols, y.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(ln_out_ptr + row * HIDDEN + cols, y.to(tl.bfloat16), mask=mask)
    else:
        tl.store(ln_out_ptr + row * HIDDEN + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 16}),
        triton.Config({'num_warps': 8}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 2}),
    ],
    key=['N_ROWS', 'HIDDEN'],
)
@triton.jit
def _embed_add_ln_1out_kernel(
    word_ids_ptr, tt_ids_ptr, pos_ids_ptr,
    word_emb_ptr, tt_emb_ptr, pos_emb_ptr,
    ln_w_ptr, ln_b_ptr,
    ln_out_ptr,
    N_ROWS, seq_len,
    HIDDEN: tl.constexpr,
    pos_batch_stride,
    eps,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // seq_len
    s = row % seq_len

    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN

    # Load indices
    word_idx = tl.load(word_ids_ptr + b * seq_len + s)
    tt_idx = tl.load(tt_ids_ptr + b * seq_len + s)
    pos_idx = tl.load(pos_ids_ptr + b * pos_batch_stride + s)

    # Load word embedding (with padding_idx=0 zeroing)
    x = tl.load(word_emb_ptr + word_idx * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(word_idx == 0, tl.zeros([BLOCK], dtype=tl.float32), x)

    # Add token type embedding
    x = x + tl.load(tt_emb_ptr + tt_idx * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)

    # Add position embedding
    x = x + tl.load(pos_emb_ptr + pos_idx * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)

    # Layer norm
    x_valid = tl.where(mask, x, 0.0)
    mean = tl.sum(x_valid, 0) / HIDDEN
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, 0) / HIDDEN
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = tl.where(mask, xc * rstd, 0.0)

    # Apply LN weight and bias
    w = tl.load(ln_w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b_v = tl.load(ln_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(mask, x_norm * w + b_v, 0.0)

    # Store LN output
    if IS_FP16:
        tl.store(ln_out_ptr + row * HIDDEN + cols, y.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(ln_out_ptr + row * HIDDEN + cols, y.to(tl.bfloat16), mask=mask)
    else:
        tl.store(ln_out_ptr + row * HIDDEN + cols, y, mask=mask)


def _get_dtype_flags(dtype):
    is_fp16 = (dtype == torch.float16)
    is_bf16 = (dtype == torch.bfloat16)
    return is_fp16, is_bf16


def fused_embed_add_ln_2out(word_ids, tt_ids, pos_ids,
                             word_emb, tt_emb, pos_emb,
                             ln_w, ln_b,
                             hidden, block_size):
    """
    Fused: 3 embedding lookups + add + layer_norm
    Returns (pre_ln, ln_out) each of shape [B, S, hidden]
    """
    B, S = word_ids.shape
    N_ROWS = B * S
    dtype = word_emb.dtype
    device = word_emb.device

    # pos_ids may be [1, S] (broadcast) or [B, S]
    pos_batch_stride = 0 if pos_ids.shape[0] == 1 else S

    pre_ln = torch.empty((N_ROWS, hidden), dtype=dtype, device=device)
    ln_out = torch.empty((N_ROWS, hidden), dtype=dtype, device=device)

    is_fp16, is_bf16 = _get_dtype_flags(dtype)

    _embed_add_ln_2out_kernel[(N_ROWS,)](
        word_ids, tt_ids, pos_ids,
        word_emb, tt_emb, pos_emb,
        ln_w, ln_b,
        pre_ln, ln_out,
        N_ROWS, S,
        hidden,
        pos_batch_stride,
        1e-12,
        is_fp16, is_bf16,
        block_size,
    )

    return pre_ln.view(B, S, hidden), ln_out.view(B, S, hidden)


def fused_embed_add_ln_1out(word_ids, tt_ids, pos_ids,
                             word_emb, tt_emb, pos_emb,
                             ln_w, ln_b,
                             hidden, block_size):
    """
    Fused: 3 embedding lookups + add + layer_norm
    Returns ln_out of shape [B, S, hidden]
    """
    B, S = word_ids.shape
    N_ROWS = B * S
    dtype = word_emb.dtype
    device = word_emb.device

    pos_batch_stride = 0 if pos_ids.shape[0] == 1 else S

    ln_out = torch.empty((N_ROWS, hidden), dtype=dtype, device=device)

    is_fp16, is_bf16 = _get_dtype_flags(dtype)

    _embed_add_ln_1out_kernel[(N_ROWS,)](
        word_ids, tt_ids, pos_ids,
        word_emb, tt_emb, pos_emb,
        ln_w, ln_b,
        ln_out,
        N_ROWS, S,
        hidden,
        pos_batch_stride,
        1e-12,
        is_fp16, is_bf16,
        block_size,
    )

    return ln_out.view(B, S, hidden)