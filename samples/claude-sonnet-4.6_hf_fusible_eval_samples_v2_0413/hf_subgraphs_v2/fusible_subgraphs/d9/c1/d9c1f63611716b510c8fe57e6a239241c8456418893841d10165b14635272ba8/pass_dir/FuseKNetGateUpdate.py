import torch
import triton
import triton.language as tl


# ============================================================
# Pattern: matches the full forward pass computation
# ============================================================
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
    linear = torch.nn.functional.linear(in_8, in_7, in_6)
    tmp_9 = torch.nn.functional.layer_norm(linear, (256,), in_3, in_2, 1e-05)
    tmp_10 = in_9.sigmoid()
    tmp_11 = tmp_9.sigmoid()
    tmp_12 = torch.nn.functional.layer_norm(in_11, (256,), in_5, in_4, 1e-05)
    tmp_13 = torch.nn.functional.layer_norm(in_10, (256,), in_1, in_0, 1e-05)
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    return (tmp_17,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11)


# ============================================================
# Kernel 1: Tiled matmul with bias  (A @ B^T + bias)
#   A  : [M, K]  (in_8 viewed as [300, 256])
#   B  : [N, K]  (in_7,  weight matrix [256, 256])
#   bias: [N]    (in_6,  [256])
#   C  : [M, N]  float32 output, consumed by Kernel 2
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_idx = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_idx = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_idx = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_mask = (m_idx[:, None] < M) & (k_idx[None, :] < K)
        a = tl.load(
            A_ptr + m_idx[:, None] * stride_am + k_idx[None, :] * stride_ak,
            mask=a_mask, other=0.0
        ).to(tl.float32)

        # Load B tile as [BLOCK_K, BLOCK_N] (transposed access):
        #   b[ki, ni] = B[n_start+ni, k_start+ki]
        #             = B_ptr + (n_start+ni)*stride_bn + (k_start+ki)*stride_bk
        b_mask = (k_idx[:, None] < K) & (n_idx[None, :] < N)
        b = tl.load(
            B_ptr + n_idx[None, :] * stride_bn + k_idx[:, None] * stride_bk,
            mask=b_mask, other=0.0
        ).to(tl.float32)

        # a: [BLOCK_M, BLOCK_K],  b: [BLOCK_K, BLOCK_N]
        acc = tl.dot(a, b, acc)

    # Add bias
    bias = tl.load(bias_ptr + n_idx, mask=(n_idx < N), other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store to float32 C
    c_mask = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    tl.store(C_ptr + m_idx[:, None] * N + n_idx[None, :], acc, mask=c_mask)


# ============================================================
# Kernel 2: Fused post-processing
#   - LayerNorm(linear_out) → sigmoid   [gate path]
#   - sigmoid(in_9)                     [input gate path]
#   - LayerNorm(in_11)                  [param_out path, unsqueeze is trivial]
#   - LayerNorm(in_10)                  [input_out path]
#   - result = gate * ln_param + sig_in9 * ln_input
#
#   Each program handles one row of D=256 elements.
#   All [300,1,256] tensors have identical stride to [300,256].
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 8}),
        triton.Config({'num_warps': 16}),
    ],
    key=['NROWS'],
)
@triton.jit
def fused_post_kernel(
    # linear output (float32 from Kernel 1) [NROWS, D]
    linear_ptr,
    # LN weights/biases for gate: in_3 (weight), in_2 (bias)
    ln_w_gate_ptr, ln_b_gate_ptr,
    # in_9: input gate — raw tensor pointer, row stride passed separately
    in9_ptr,
    # LN weights/biases for param_out: in_5 (weight), in_4 (bias)
    ln_w_param_ptr, ln_b_param_ptr,
    # in_11: param_out — raw tensor pointer, row stride passed separately
    in11_ptr,
    # LN weights/biases for input_out: in_1 (weight), in_0 (bias)
    ln_w_input_ptr, ln_b_input_ptr,
    # in_10: input_out — raw tensor pointer, row stride passed separately
    in10_ptr,
    # output tensor pointer with row stride
    out_ptr,
    NROWS,
    D: tl.constexpr,
    stride_in9,
    stride_in11,
    stride_in10,
    stride_out,
):
    row = tl.program_id(0)
    if row >= NROWS:
        return
    offsets = tl.arange(0, D)  # D=256, power of 2

    # Row bases using passed strides
    lin_base  = row * D            # linear_out is always [M, D] contiguous
    in9_base  = row * stride_in9
    in11_base = row * stride_in11
    in10_base = row * stride_in10
    out_base  = row * stride_out

    # ---- Gate path: LayerNorm(linear_out) -> sigmoid ----
    x = tl.load(linear_ptr + lin_base + offsets)  # float32 already
    mean_x = tl.sum(x) / D
    x_c = x - mean_x
    var_x = tl.sum(x_c * x_c) / D
    inv_std_x = tl.rsqrt(var_x + 1e-5)
    w_gate = tl.load(ln_w_gate_ptr + offsets).to(tl.float32)
    b_gate = tl.load(ln_b_gate_ptr + offsets).to(tl.float32)
    tmp9 = x_c * inv_std_x * w_gate + b_gate
    tmp11 = 1.0 / (1.0 + tl.exp(-tmp9))   # sigmoid(tmp9) = gate

    # ---- Input gate path: sigmoid(in_9) ----
    in9 = tl.load(in9_ptr + in9_base + offsets).to(tl.float32)
    tmp10 = 1.0 / (1.0 + tl.exp(-in9))

    # ---- Param path: LayerNorm(in_11) — unsqueeze(-2) is a no-op ----
    in11 = tl.load(in11_ptr + in11_base + offsets).to(tl.float32)
    mean11 = tl.sum(in11) / D
    diff11 = in11 - mean11
    var11 = tl.sum(diff11 * diff11) / D
    inv_std11 = tl.rsqrt(var11 + 1e-5)
    w_param = tl.load(ln_w_param_ptr + offsets).to(tl.float32)
    b_param = tl.load(ln_b_param_ptr + offsets).to(tl.float32)
    tmp12 = diff11 * inv_std11 * w_param + b_param

    # ---- Input out path: LayerNorm(in_10) ----
    in10 = tl.load(in10_ptr + in10_base + offsets).to(tl.float32)
    mean10 = tl.sum(in10) / D
    diff10 = in10 - mean10
    var10 = tl.sum(diff10 * diff10) / D
    inv_std10 = tl.rsqrt(var10 + 1e-5)
    w_input = tl.load(ln_w_input_ptr + offsets).to(tl.float32)
    b_input = tl.load(ln_b_input_ptr + offsets).to(tl.float32)
    tmp13 = diff10 * inv_std10 * w_input + b_input

    # ---- Final gating: tmp11*tmp12 + tmp10*tmp13 ----
    result = tmp11 * tmp12 + tmp10 * tmp13

    # Store — Triton auto-casts float32→output dtype (fp16/bf16/fp32)
    tl.store(out_ptr + out_base + offsets, result)


# ============================================================
# Wrapper: called by the framework in place of the matched pattern
# ============================================================
@torch.fx.wrap
def fused_knet_gate_update(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
    # Shapes:
    #  in_8  [300, 1, 256]  gate_feats        strides (256, 256, 1)
    #  in_7  [256, 256]     update_gate weight strides (256, 1)
    #  in_6  [256]          update_gate bias
    #  in_3  [256]          norm_in weight (for LN of linear output)
    #  in_2  [256]          norm_in bias
    #  in_9  [300, 1, 256]  input_gate        strides (256, 256, 1)
    #  in_5  [256]          norm_out weight
    #  in_4  [256]          norm_out bias
    #  in_11 [300, 256]     param_out         strides (256, 1)
    #  in_1  [256]          input_norm_out weight
    #  in_0  [256]          input_norm_out bias
    #  in_10 [300, 1, 256]  input_out         strides (256, 256, 1)

    M = in_8.shape[0]           # 300
    D = 256
    N = D                       # output features = 256
    K = D                       # input features  = 256
    device = in_8.device

    # ---- Step 1: Linear (matmul + bias) → float32 intermediate ----
    # in_8 is [300,1,256]: use stride(0)=256 as row stride, stride(2)=1 as col stride
    # in_7 is [256,256]: standard strides
    linear_out = torch.empty((M, N), dtype=torch.float32, device=device)

    def linear_grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    matmul_bias_kernel[linear_grid](
        in_8, in_7, in_6, linear_out,
        M, N, K,
        in_8.stride(0), in_8.stride(2),   # row-stride=256, col-stride=1 (skip trivial dim)
        in_7.stride(0), in_7.stride(1),   # weight strides (256, 1)
    )

    # ---- Step 2: Fused post-processing ----
    # Output has same shape/dtype as in_8: [300, 1, 256]
    # Triton will auto-cast float32 result to in_8.dtype when storing
    out = torch.empty_like(in_8)   # [300, 1, 256], same dtype as inputs

    fused_post_kernel[(M,)](
        linear_out,
        in_3, in_2,         # LN gate: weight=in_3, bias=in_2
        in_9,
        in_5, in_4,         # LN param: weight=in_5, bias=in_4
        in_11,
        in_1, in_0,         # LN input: weight=in_1, bias=in_0
        in_10,
        out,
        NROWS=M,
        D=D,
        # Row strides (skip the trivial size-1 middle dimension)
        stride_in9=in_9.stride(0),
        stride_in11=in_11.stride(0),
        stride_in10=in_10.stride(0),
        stride_out=out.stride(0),
    )

    return (out,)


def replacement_func():
    return fused_knet_gate_update