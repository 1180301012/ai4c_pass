import torch
import triton
import triton.language as tl


@triton.jit
def fused_mh_selfattn_kernel(
    # Input projection: x -> [Q,K,V] (combined)
    x_ptr, x_stride_s, x_stride_d,
    in_proj_weight_ptr, in_proj_weight_stride_i, in_proj_weight_stride_k,
    in_proj_bias_ptr,
    # Attention output projection
    out_proj_weight_ptr, out_proj_weight_stride_o, out_proj_weight_stride_i,
    out_proj_bias_ptr,
    # Attention output
    out_ptr, out_stride_s, out_stride_d,
    # Fixed dimensions
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    # Tile sizes
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # Each program handles one head and one block of query positions
    pid_q = tl.program_id(0)   # which query-block
    pid_h = tl.program_id(1)   # which head

    q_start   = pid_q * BLOCK_Q
    head_off  = pid_h * HEAD_DIM

    offs_q = q_start + tl.arange(0, BLOCK_Q)   # [BLOCK_Q]
    offs_d = tl.arange(0, HEAD_DIM)             # [HEAD_DIM]

    # ------------------------------------------------------------------ #
    # Load input token embeddings: x[s, 0, d]
    # Contiguous layout: element (s, 0, d)  ->  base + s*x_stride_s + d*x_stride_d
    # ------------------------------------------------------------------ #
    x = tl.load(
        x_ptr + offs_q[:, None] * x_stride_s + offs_d[None, :] * x_stride_d,
        mask=offs_q[:, None] < SEQ_LEN,
        other=0.0,
    ).to(tl.float32)                            # [BLOCK_Q, HEAD_DIM]

    # ------------------------------------------------------------------ #
    # Compute Q, K, V  via  x @ W_i + b_i   (self-attention: Q=K=V=x)
    # W_i : [3*HEAD_DIM, EMBED_DIM],  b_i : [3*HEAD_DIM]
    # Split:  Q = slice [0  : HEAD_DIM]
    #         K = slice [HEAD_DIM : 2*HEAD_DIM]
    #         V = slice [2*HEAD_DIM : 3*HEAD_DIM]
    # ------------------------------------------------------------------ #
    w_shape = 3 * HEAD_DIM                      # scalar

    qkv = tl.zeros([BLOCK_Q, w_shape], dtype=tl.float32)

    for d_idx in range(0, HEAD_DIM, 16):
        offs_d16 = d_idx + tl.arange(0, 16)
        # Weight rows for Q
        wq = tl.load(in_proj_weight_ptr + offs_d16[:, None] * in_proj_weight_stride_i
                     + offs_d[None, :] * in_proj_weight_stride_k)
        qkv[:, :HEAD_DIM] += tl.dot(x, wq, allow_tf32=False)
        # Weight rows for K
        wk = tl.load(in_proj_weight_ptr + (HEAD_DIM + offs_d16)[:, None] * in_proj_weight_stride_i
                     + offs_d[None, :] * in_proj_weight_stride_k)
        qkv[:, HEAD_DIM:2 * HEAD_DIM] += tl.dot(x, wk, allow_tf32=False)
        # Weight rows for V
        wv = tl.load(in_proj_weight_ptr + (2 * HEAD_DIM + offs_d16)[:, None] * in_proj_weight_stride_i
                     + offs_d[None, :] * in_proj_weight_stride_k)
        qkv[:, 2 * HEAD_DIM:] += tl.dot(x, wv, allow_tf32=False)

    qkv = qkv + in_proj_bias_ptr[None, :]       # broadcast bias

    q = qkv[:, :HEAD_DIM]                       # [BLOCK_Q, HEAD_DIM]
    k = qkv[:, HEAD_DIM:2 * HEAD_DIM]           # [BLOCK_Q, HEAD_DIM]
    v = qkv[:, 2 * HEAD_DIM:]                   # [BLOCK_Q, HEAD_DIM]

    # ------------------------------------------------------------------ #
    # Compute attention scores  Q @ K^T / sqrt(D)
    # k is [BLOCK_Q, HEAD_DIM] -> transpose to [HEAD_DIM, BLOCK_Q]
    # ------------------------------------------------------------------ #
    scale = 1.0 / (HEAD_DIM ** 0.5)
    qk = tl.zeros([BLOCK_Q, BLOCK_Q], dtype=tl.float32)

    for k_start in range(0, BLOCK_Q, 16):
        offs_k16 = k_start + tl.arange(0, 16)
        qk += tl.dot(q, tl.trans(k[offs_k16, :]), allow_tf32=False) * scale

    # k_len = SEQ_LEN  (self-attention over full sequence)
    attn = tl.where(offs_q[None, :] < SEQ_LEN, qk, float('-inf'))

    # Softmax over key dimension
    attn_max = tl.max(attn, axis=1)             # [BLOCK_Q]
    attn = tl.exp(attn - attn_max[:, None])
    attn_sum = tl.sum(attn, axis=1)             # [BLOCK_Q]
    attn = attn / attn_sum[:, None]

    # ------------------------------------------------------------------ #
    # Weighted sum with V  →  [BLOCK_Q, HEAD_DIM]
    # ------------------------------------------------------------------ #
    out = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    for v_start in range(0, BLOCK_Q, 16):
        offs_v16 = v_start + tl.arange(0, 16)
        out += tl.dot(attn[:, :HEAD_DIM], v[offs_v16, :], allow_tf32=False)

    # ------------------------------------------------------------------ #
    # Output projection:  w_out[h*D:(h+1)*D]  and  bias[h*D:(h+1)*D]
    # ------------------------------------------------------------------ #
    w_out = tl.load(
        out_proj_weight_ptr + head_off + offs_d[None, :] * out_proj_weight_stride_i
    )                                            # [1, HEAD_DIM]

    b_out = tl.load(out_proj_bias_ptr + head_off + offs_d)  # [HEAD_DIM]

    result = tl.sum(out * w_out, axis=1) + b_out         # [BLOCK_Q]

    # Cast result to output dtype and store
    if IS_FP16:
        result = result.to(tl.float16)
    elif IS_BF16:
        result = result.to(tl.bfloat16)
    # else float32 (already correct)

    tl.store(
        out_ptr + offs_q * out_stride_s + offs_d[None, :] * out_stride_d,
        result,
        mask=offs_q[:, None] < SEQ_LEN,
    )


@torch.fx.wrap
def fused_mh_selfattn(in_0, in_1, in_2, in_3, in_4):
    """
    Fused Multi-Head Self-Attention + Input/Output Projection.

    in_0 : out_proj_bias   [embed_dim]
    in_1 : out_proj_weight [embed_dim, embed_dim]
    in_2 : in_proj_bias    [3 * head_dim]
    in_3 : in_proj_weight  [3 * head_dim, embed_dim]
    in_4 : input           [seq_len, 1, embed_dim]
    """
    seq_len  = in_4.shape[0]
    embed_dim = in_4.shape[2]
    num_heads = 8
    head_dim  = embed_dim // num_heads   # 64

    is_fp16 = (in_4.dtype == torch.float16)
    is_bf16 = (in_4.dtype == torch.bfloat16)

    out = torch.empty(seq_len, 1, embed_dim, device=in_4.device, dtype=in_4.dtype)

    grid = (triton.cdiv(seq_len, 16), num_heads)

    fused_mh_selfattn_kernel[grid](
        in_4,
        in_4.stride(0), in_4.stride(2),
        in_3, in_3.stride(0), in_3.stride(1),
        in_2,
        in_1, in_1.stride(0), in_1.stride(1),
        in_0,
        out, out.stride(0), out.stride(2),
        N_HEADS=num_heads,
        HEAD_DIM=head_dim,
        SEQ_LEN=seq_len,
        BLOCK_Q=16,
        BLOCK_K=16,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )

    return out


# ------------------------------------------------------------------ #
# Pattern / replacement hooks required by the AI4C pass framework
# ------------------------------------------------------------------ #

def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match:  multi_head_attention_forward (self-attention) +
            getitem[0] + two no-op dropout calls

    Argument order mirrors model.py exactly:
      in_0 = out_proj_bias, in_1 = out_proj_weight,
      in_2 = in_proj_bias,  in_3 = in_proj_weight,  in_4 = input
    """
    result = torch.nn.functional.multi_head_attention_forward(
        in_4, in_4, in_4, 512, 8, in_3, in_2, None, None, False, 0.0, in_1, in_0,
        training=False, key_padding_mask=None, need_weights=True,
        attn_mask=None, average_attn_weights=True, is_causal=False,
    )
    attn_out = result[0]
    drop1 = torch.nn.functional.dropout(attn_out, 0.0, False, False)
    drop2 = torch.nn.functional.dropout(drop1, 0.0, False, False)
    return (drop2,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_mh_selfattn