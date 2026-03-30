"""
Full fusion pass: Token Embedding + Scale + Position Embedding + Add + LayerNorm + Dropout(no-op)

The graph does:
  1. token_emb = embedding(in_4, in_1, padding_idx=1)  -> [1, 1, 256]
  2. scaled    = token_emb * 16.0
  3. idx       = arange(0,1) -> expand(1,-1) -> +2  => constant [[2]]
  4. pos_emb   = embedding([[2]], in_0)               -> [1, 1, 256]
  5. added     = scaled + pos_emb
  6. normed    = layer_norm(added, (256,), in_3, in_2, eps=1e-5)
  7. out       = dropout(normed, p=0.1, training=False)  => identity

Fuse all into one Triton kernel that:
  - reads the token id from in_4
  - loads token embedding row * 16
  - loads positional embedding row 2 (always)
  - adds them
  - computes layer norm in fp32
  - writes result in original dtype (bf16 or fp16)
"""

import torch
import triton
import triton.language as tl
from torch import device


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
    ],
    key=["hidden_size"],
)
@triton.jit
def full_fused_embedding_layernorm_kernel(
    in4_ptr,       # token ids  [1, 1]  int64
    in1_ptr,       # token emb  [vocab, 256]  fp16/bf16
    in0_ptr,       # pos   emb  [514, 256]    fp16/bf16
    in3_ptr,       # ln weight  [256]          fp16/bf16
    in2_ptr,       # ln bias    [256]          fp16/bf16
    out_ptr,       # output     [1, 1, 256]    fp16/bf16
    hidden_size: tl.constexpr,
    scale: tl.constexpr,     # 16.0
    pos_idx: tl.constexpr,   # 2
    eps: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program handles all 256 elements (single row)
    cols = tl.arange(0, BLOCK_SIZE)

    # ---- 1. load token id (scalar int64) ----
    token_id = tl.load(in4_ptr).to(tl.int64)

    # ---- 2. load token embedding * scale ----
    tok_off = token_id * hidden_size + cols
    tok_emb = tl.load(in1_ptr + tok_off).to(tl.float32) * scale

    # ---- 3. load position embedding (always row pos_idx=2) ----
    pos_off = pos_idx * hidden_size + cols
    pos_emb = tl.load(in0_ptr + pos_off).to(tl.float32)

    # ---- 4. add ----
    val = tok_emb + pos_emb

    # ---- 5. layer norm ----
    mean = tl.sum(val, axis=0) / hidden_size
    diff = val - mean
    var  = tl.sum(diff * diff, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)
    norm = diff * rstd

    # ---- 6. affine: weight * norm + bias ----
    w      = tl.load(in3_ptr + cols).to(tl.float32)
    b      = tl.load(in2_ptr + cols).to(tl.float32)
    result = norm * w + b

    # ---- 7. store (dropout is identity at training=False) ----
    if IS_BF16:
        tl.store(out_ptr + cols, result.to(tl.bfloat16))
    else:
        tl.store(out_ptr + cols, result.to(tl.float16))


@torch.fx.wrap
def full_fused_embedding_layernorm(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : position embedding table [514, 256]  fp16/bf16
    in_1 : token    embedding table [vocab, 256] fp16/bf16
    in_2 : layer-norm bias          [256]        fp16/bf16
    in_3 : layer-norm weight        [256]        fp16/bf16
    in_4 : token ids                [1, 1]       int64
    """
    is_bf16 = (in_1.dtype == torch.bfloat16)
    out = torch.empty(1, 1, 256, dtype=in_1.dtype, device=in_1.device)

    full_fused_embedding_layernorm_kernel[(1,)](
        in_4,          # token ids
        in_1,          # token embedding table
        in_0,          # position embedding table
        in_3,          # layer norm weight
        in_2,          # layer norm bias
        out,           # output
        hidden_size=256,
        scale=16.0,
        pos_idx=2,
        eps=1e-5,
        IS_BF16=is_bf16,
    )
    return out


# -------------------------------------------------------------------------
# Pattern: mirrors model.py forward() exactly
# -------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4  = torch.nn.functional.embedding(in_4, in_1, 1, None, 2.0, False, False)
    tmp_5  = tmp_4 * 16.0
    tmp_6  = torch.arange(0, 1, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_7  = tmp_6.expand(1, -1)
    tmp_8  = tmp_7 + 2
    tmp_9  = torch.nn.functional.embedding(tmp_8, in_0, None, None, 2.0, False, False)
    tmp_10 = tmp_5 + tmp_9
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (256,), in_3, in_2, 1e-05)
    tmp_12 = torch.nn.functional.dropout(tmp_11, p=0.1, training=False)
    return tmp_12


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return full_fused_embedding_layernorm