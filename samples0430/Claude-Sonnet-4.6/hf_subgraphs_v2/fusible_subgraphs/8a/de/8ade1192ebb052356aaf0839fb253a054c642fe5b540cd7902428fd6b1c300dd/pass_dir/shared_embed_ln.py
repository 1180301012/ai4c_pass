"""
Shared Triton kernel for fused embedding lookup + residual add + layer norm.
Imported by FuseEmbedAddLayerNorm_H*.py pass files.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def embed_add_ln_kernel(
    in0_ptr,   # [B, S, H] - input residual (bf16 or fp16)
    in1_ptr,   # [V, H]    - embedding weight table (bf16 or fp16)
    in2_ptr,   # [H]       - layer norm bias (bf16 or fp16)
    in3_ptr,   # [H]       - layer norm weight (bf16 or fp16)
    in4_ptr,   # [S]       - position indices (int64)
    out_ptr,   # [B, S, H] - output (same dtype as in0)
    stride_b,  # stride for batch dim of in0
    stride_s,  # stride for seq dim of in0
    S,         # sequence length
    H,         # hidden dimension
    eps,       # layer norm epsilon
    BLOCK_H: tl.constexpr,  # tile size >= H (power of 2)
    IS_BF16: tl.constexpr,  # True => bfloat16, False => float16
):
    # Each program handles one (batch, seq) row
    row_id = tl.program_id(0)
    b = row_id // S
    s = row_id % S

    # Load position index and compute embedding table row
    pos = tl.load(in4_ptr + s)   # int64 scalar
    emb_idx = pos + 2             # int64, shifted by +2

    # Element offsets within hidden dim
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    # Load input residual: in_0[b, s, :]
    base = b * stride_b + s * stride_s
    x0 = tl.load(in0_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # Load position embedding: in_1[emb_idx, :]
    x1 = tl.load(in1_ptr + emb_idx * H + offs, mask=mask, other=0.0).to(tl.float32)

    # Fused residual add
    x = x0 + x1

    # ---- Layer Norm ----
    # Mean (mask out padding positions)
    x_safe = tl.where(mask, x, 0.0)
    mean = tl.sum(x_safe, axis=0) / H

    # Variance
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / H
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_norm = (x - mean) * inv_std

    # Scale and bias
    ln_w = tl.load(in3_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    ln_b = tl.load(in2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * ln_w + ln_b

    # Store back in original dtype (dropout training=False is identity)
    if IS_BF16:
        tl.store(out_ptr + base + offs, out.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + base + offs, out.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_embed_add_layernorm(in_0, in_1, in_2, in_3, in_4):
    """
    Fused: embedding_lookup(in_4+2, in_1) + in_0, then layer_norm with in_3/in_2.
    Dropout(training=False) is identity and is skipped.
    """
    B, S, H = in_0.shape
    out = torch.empty_like(in_0)

    # BLOCK_H must be a power-of-2 >= H so the entire row fits in one tile
    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_H = max(BLOCK_H, 16)

    IS_BF16 = (in_0.dtype == torch.bfloat16)

    # Heuristic num_warps based on tile width
    if BLOCK_H >= 512:
        num_warps = 8
    elif BLOCK_H >= 128:
        num_warps = 4
    elif BLOCK_H >= 32:
        num_warps = 2
    else:
        num_warps = 1

    grid = (B * S,)
    embed_add_ln_kernel[grid](
        in_0, in_1, in_2, in_3, in_4, out,
        in_0.stride(0), in_0.stride(1),
        S, H, 1e-5,
        BLOCK_H=BLOCK_H,
        IS_BF16=IS_BF16,
        num_warps=num_warps,
    )
    return out