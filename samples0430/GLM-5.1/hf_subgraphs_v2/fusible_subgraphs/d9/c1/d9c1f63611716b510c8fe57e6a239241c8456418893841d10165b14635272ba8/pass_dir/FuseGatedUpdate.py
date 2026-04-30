import torch
import triton
import triton.language as tl


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


# ===== Triton Matmul Kernel (for F.linear) =====
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Tiled matmul kernel for F.linear: C = A @ B^T + bias.
    A is in_8 [M, K], B^T is in_7.T [K, N], C is linear result [M, N].
    We load B in transposed order from in_7: B[k,n] = in_7[n,k].
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        # A[m, k] = in_8[m, 0, k] (shape [M, 1, K] treated as [M, K])
        a = tl.load(a_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak,
                    mask=m_mask[:, None], other=0.0)

        # Load B tile in transposed order: [BLOCK_K, BLOCK_N]
        # B[k, n] = in_7[n, k], so stride_bk = stride_in7_col, stride_bn = stride_in7_row
        b = tl.load(b_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn,
                    mask=n_mask[None, :], other=0.0)

        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store result - Triton handles dtype conversion automatically
    tl.store(c_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn,
             acc, mask=m_mask[:, None] & n_mask[None, :])


# ===== Triton Post-Processing Kernel =====
@triton.jit
def post_process_kernel(
    linear_ptr,
    ln1_w_ptr, ln1_b_ptr,      # in_3 (weight), in_2 (bias) for layer_norm on linear
    in_9_ptr,
    in_11_ptr, ln2_w_ptr, ln2_b_ptr,  # in_5 (weight), in_4 (bias) for layer_norm on in_11
    in_10_ptr, ln3_w_ptr, ln3_b_ptr,  # in_1 (weight), in_0 (bias) for layer_norm on in_10
    out_ptr,
    N_ROWS, N_COLS,
    stride_linear_0, stride_linear_2,
    stride_in9_0, stride_in9_2,
    stride_in11_0, stride_in11_1,
    stride_in10_0, stride_in10_2,
    stride_out_0, stride_out_2,
    BLOCK_SIZE: tl.constexpr,
):
    """Post-processing kernel: applies layer_norms, sigmoids, multiplications, and addition.
    Each program handles one full row (all N_COLS elements).
    """
    row_idx = tl.program_id(0)
    if row_idx >= N_ROWS:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    # ===== Layer Norm 1: on linear result =====
    linear = tl.load(linear_ptr + row_idx * stride_linear_0 + cols * stride_linear_2,
                     mask=mask, other=0.0).to(tl.float32)

    mean1 = tl.sum(linear, axis=0) / N_COLS
    diff1 = linear - mean1
    var1 = tl.sum(diff1 * diff1, axis=0) / N_COLS
    rstd1 = 1.0 / tl.sqrt(var1 + 1e-05)
    tmp_9 = diff1 * rstd1

    ln1_w = tl.load(ln1_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ln1_b = tl.load(ln1_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tmp_9 = tmp_9 * ln1_w + ln1_b

    # sigmoid(tmp_9) -> tmp_11
    tmp_11 = tl.sigmoid(tmp_9)

    # ===== sigmoid(in_9) -> tmp_10 =====
    in_9_row = tl.load(in_9_ptr + row_idx * stride_in9_0 + cols * stride_in9_2,
                       mask=mask, other=0.0).to(tl.float32)
    tmp_10 = tl.sigmoid(in_9_row)

    # ===== Layer Norm 2: on in_11 -> tmp_12 =====
    in_11_row = tl.load(in_11_ptr + row_idx * stride_in11_0 + cols * stride_in11_1,
                        mask=mask, other=0.0).to(tl.float32)

    mean2 = tl.sum(in_11_row, axis=0) / N_COLS
    diff2 = in_11_row - mean2
    var2 = tl.sum(diff2 * diff2, axis=0) / N_COLS
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-05)
    tmp_12 = diff2 * rstd2

    ln2_w = tl.load(ln2_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ln2_b = tl.load(ln2_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tmp_12 = tmp_12 * ln2_w + ln2_b

    # ===== Layer Norm 3: on in_10 -> tmp_13 =====
    in_10_row = tl.load(in_10_ptr + row_idx * stride_in10_0 + cols * stride_in10_2,
                        mask=mask, other=0.0).to(tl.float32)

    mean3 = tl.sum(in_10_row, axis=0) / N_COLS
    diff3 = in_10_row - mean3
    var3 = tl.sum(diff3 * diff3, axis=0) / N_COLS
    rstd3 = 1.0 / tl.sqrt(var3 + 1e-05)
    tmp_13 = diff3 * rstd3

    ln3_w = tl.load(ln3_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ln3_b = tl.load(ln3_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tmp_13 = tmp_13 * ln3_w + ln3_b

    # ===== Multiplications and addition =====
    # tmp_14 = tmp_12.unsqueeze(-2) -> just a view change, no computation
    # tmp_15 = tmp_11 * tmp_14 = tmp_11 * tmp_12
    # tmp_16 = tmp_10 * tmp_13
    # tmp_17 = tmp_15 + tmp_16
    tmp_15 = tmp_11 * tmp_12
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16

    # ===== Store result =====
    tl.store(out_ptr + row_idx * stride_out_0 + cols * stride_out_2, tmp_17, mask=mask)


# ===== Kernel Wrapper =====
@torch.fx.wrap
def fused_gate_update(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
    M = in_8.shape[0]       # 300 (number of rows)
    K = in_8.shape[-1]      # 256 (input dimension)
    N = in_7.shape[0]       # 256 (output dimension, same as K for this model)

    # Allocate buffer - used for both linear result and final output
    # Shape [M, 1, N] matches the original computation's output shape
    buffer = torch.empty((M, 1, N), dtype=in_8.dtype, device=in_8.device)

    # Step 1: Compute linear = in_8 @ in_7.T + in_6
    # A = in_8 [M, 1, K] treated as [M, K]
    # B^T = in_7.T, but we load B in transposed order from in_7
    # So stride_bk = in_7.stride(1) (col stride = K dim for B)
    # stride_bn = in_7.stride(0) (row stride = N dim for B)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    linear_kernel[grid](
        a_ptr=in_8, b_ptr=in_7, c_ptr=buffer, bias_ptr=in_6,
        M=M, N=N, K=K,
        stride_am=in_8.stride(0), stride_ak=in_8.stride(2),
        stride_bk=in_7.stride(1), stride_bn=in_7.stride(0),
        stride_cm=buffer.stride(0), stride_cn=buffer.stride(2),
    )

    # Step 2: Post-processing (layer_norms, sigmoids, multiplications, addition)
    N_ROWS = M
    N_COLS = N  # 256
    grid2 = (N_ROWS,)
    post_process_kernel[grid2](
        linear_ptr=buffer,
        ln1_w_ptr=in_3, ln1_b_ptr=in_2,
        in_9_ptr=in_9,
        in_11_ptr=in_11, ln2_w_ptr=in_5, ln2_b_ptr=in_4,
        in_10_ptr=in_10, ln3_w_ptr=in_1, ln3_b_ptr=in_0,
        out_ptr=buffer,
        N_ROWS=N_ROWS, N_COLS=N_COLS,
        stride_linear_0=buffer.stride(0), stride_linear_2=buffer.stride(2),
        stride_in9_0=in_9.stride(0), stride_in9_2=in_9.stride(2),
        stride_in11_0=in_11.stride(0), stride_in11_1=in_11.stride(1),
        stride_in10_0=in_10.stride(0), stride_in10_2=in_10.stride(2),
        stride_out_0=buffer.stride(0), stride_out_2=buffer.stride(2),
        BLOCK_SIZE=256,
    )

    return (buffer,)


def replacement_func():
    return fused_gate_update