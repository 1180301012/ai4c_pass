import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = torch.nn.functional.multi_head_attention_forward(
        in_4, in_4, in_4, 512, 8, in_3, in_2, None, None, False, 0.0,
        in_1, in_0, training=False, key_padding_mask=None, need_weights=True,
        attn_mask=None, average_attn_weights=True, is_causal=False
    )
    tmp_5 = tmp_4[0]
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Compute C = A @ B^T + bias
    A: [M, K], B: [N, K], bias: [N], C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        # Load A[rm, rk]: [BLOCK_M, BLOCK_K]
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                    mask=a_mask, other=0.0)

        # Load B^T[rk, rn]: [BLOCK_K, BLOCK_N]
        bt_mask = (rk[:, None] < K) & (rn[None, :] < N)
        bt = tl.load(B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                     mask=bt_mask, other=0.0)

        acc += tl.dot(a, bt)

    # Add bias
    bias_mask = rn < N
    bias = tl.load(bias_ptr + rn, mask=bias_mask, other=0.0)
    acc += bias[None, :]

    # Store
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             acc.to(C_ptr.dtype.element_ty), mask=c_mask)


@triton.jit
def fused_attention_kernel(
    QKV_ptr, Out_ptr,
    seq_len,
    scale,
    k_offset,
    v_offset,
    stride_s,
    stride_h,
    stride_os,
    stride_oh,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Fused multi-head self-attention kernel.
    QKV: [seq_len, 3*embed_dim] containing Q, K, V concatenated along last dim
    Out: [seq_len, embed_dim] attention output
    """
    head_idx = tl.program_id(0)
    q_block_idx = tl.program_id(1)

    q_start = q_block_idx * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, HEAD_DIM)

    # Base pointers for Q, K, V for this head
    Q_base = QKV_ptr + head_idx * stride_h
    K_base = QKV_ptr + k_offset + head_idx * stride_h
    V_base = QKV_ptr + v_offset + head_idx * stride_h
    O_base = Out_ptr + head_idx * stride_oh

    # Load Q: [BLOCK_Q, HEAD_DIM]
    q_mask = offs_q[:, None] < seq_len
    q = tl.load(Q_base + offs_q[:, None] * stride_s + offs_d[None, :],
                mask=q_mask, other=0.0)

    # Load K^T: [HEAD_DIM, BLOCK_K]
    kt_mask = offs_k[None, :] < seq_len
    kt = tl.load(K_base + offs_d[:, None] + offs_k[None, :] * stride_s,
                 mask=kt_mask, other=0.0)

    # Compute scores: [BLOCK_Q, BLOCK_K]
    scores = tl.dot(q, kt)
    scores = scores * scale

    # Mask invalid positions
    score_mask = (offs_q[:, None] < seq_len) & (offs_k[None, :] < seq_len)
    scores = tl.where(score_mask, scores, float('-inf'))

    # Softmax
    max_scores = tl.max(scores, axis=1)  # [BLOCK_Q]
    scores = scores - max_scores[:, None]
    exp_scores = tl.exp(scores)
    exp_scores = tl.where(score_mask, exp_scores, 0.0)
    sum_exp = tl.sum(exp_scores, axis=1)  # [BLOCK_Q]
    sum_exp = tl.where(sum_exp > 0.0, sum_exp, 1.0)
    attn_weights = exp_scores / sum_exp[:, None]

    # Load V: [BLOCK_K, HEAD_DIM]
    v_mask = offs_k[:, None] < seq_len
    v = tl.load(V_base + offs_k[:, None] * stride_s + offs_d[None, :],
                mask=v_mask, other=0.0)

    # Cast attn_weights to match V dtype for tl.dot
    attn_weights = attn_weights.to(v.dtype)

    # Compute output: [BLOCK_Q, HEAD_DIM]
    output = tl.dot(attn_weights, v)

    # Store output
    o_mask = offs_q[:, None] < seq_len
    tl.store(O_base + offs_q[:, None] * stride_os + offs_d[None, :],
             output.to(Out_ptr.dtype.element_ty), mask=o_mask)


@torch.fx.wrap
def optimized_mha(out_proj_bias, out_proj_weight, in_proj_bias, in_proj_weight, x):
    """Optimized multi-head attention replacing MHA + dropout chain."""
    seq_len = x.shape[0]  # 150
    embed_dim = 512
    num_heads = 8
    head_dim = 64

    # Step 1: In-projection
    # x: [seq_len, 1, 512] treated as [seq_len, 512]
    # in_proj_weight: [1536, 512]
    # in_proj_bias: [1536]
    # Result: qkv [seq_len, 1536]
    qkv = torch.empty((seq_len, 3 * embed_dim), dtype=x.dtype, device=x.device)
    M_in, N_in, K_in = seq_len, 3 * embed_dim, embed_dim
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 64
    grid_in = ((M_in + BLOCK_M - 1) // BLOCK_M, (N_in + BLOCK_N - 1) // BLOCK_N)
    matmul_bias_kernel[grid_in](
        x, in_proj_weight, in_proj_bias, qkv,
        M_in, N_in, K_in,
        embed_dim, 1,         # A strides: x as [seq_len, 512]
        embed_dim, 1,         # B strides: in_proj_weight [1536, 512]
        3 * embed_dim, 1,     # C strides: qkv [seq_len, 1536]
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Step 2: Fused attention
    # Q, K, V are views into qkv with offsets 0, 512, 1024
    # Interpreted as [num_heads, seq_len, head_dim] with strides:
    #   head: head_dim=64, seq: 3*embed_dim=1536, dim: 1
    attn_out = torch.empty((seq_len, embed_dim), dtype=x.dtype, device=x.device)
    scale = 0.125  # 1/sqrt(64) = 1/8
    BLOCK_Q_ATT, BLOCK_K_ATT = 32, 256
    grid_attn = (num_heads, (seq_len + BLOCK_Q_ATT - 1) // BLOCK_Q_ATT)
    fused_attention_kernel[grid_attn](
        qkv, attn_out,
        seq_len, scale,
        embed_dim,            # k_offset = 512 elements
        2 * embed_dim,        # v_offset = 1024 elements
        3 * embed_dim,        # stride_s = 1536 (seq stride in qkv)
        head_dim,             # stride_h = 64 (head stride in qkv)
        embed_dim,            # stride_os = 512 (seq stride in output)
        head_dim,             # stride_oh = 64 (head stride in output)
        BLOCK_Q=BLOCK_Q_ATT, BLOCK_K=BLOCK_K_ATT, HEAD_DIM=head_dim,
    )

    # Step 3: Out-projection
    # attn_out: [seq_len, 512]
    # out_proj_weight: [512, 512]
    # out_proj_bias: [512]
    # Result: [seq_len, 1, 512]
    output = torch.empty((seq_len, 1, embed_dim), dtype=x.dtype, device=x.device)
    M_out, N_out, K_out = seq_len, embed_dim, embed_dim
    grid_out = ((M_out + BLOCK_M - 1) // BLOCK_M, (N_out + BLOCK_N - 1) // BLOCK_N)
    matmul_bias_kernel[grid_out](
        attn_out, out_proj_weight, out_proj_bias, output,
        M_out, N_out, K_out,
        embed_dim, 1,         # A strides: attn_out [seq_len, 512]
        embed_dim, 1,         # B strides: out_proj_weight [512, 512]
        embed_dim, 1,         # C strides: output as [seq_len, 512] (= [seq_len, 1, 512])
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return output


def replacement_func():
    return optimized_mha