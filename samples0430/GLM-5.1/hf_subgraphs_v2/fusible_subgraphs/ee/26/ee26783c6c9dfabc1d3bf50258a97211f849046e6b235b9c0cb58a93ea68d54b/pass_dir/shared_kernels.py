import torch
import triton
import triton.language as tl


@triton.jit
def matmul1_kernel(
    in1_ptr, stride_in1_0, stride_in1_1, stride_in1_2, stride_in1_3,
    in3_ptr, stride_in3_0, stride_in3_1,
    out_ptr, stride_out_0, stride_out_1, stride_out_2, stride_out_3,
    B: tl.constexpr, HW: tl.constexpr, D_INNER: tl.constexpr, L: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    DTYPE_CODE: tl.constexpr,
):
    """Kernel for computing in_1 @ in_3 where in_1 is [B, HW, HW, D_INNER] and in_3 is [D_INNER, L]."""
    batch_id = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    b = batch_id // HW
    h = batch_id % HW

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < HW
    n_mask = n_offsets < L

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, D_INNER, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < D_INNER

        # Load in_1[b, h, m, k]
        a_ptrs = in1_ptr + b * stride_in1_0 + h * stride_in1_1 + m_offsets[:, None] * stride_in1_2 + k_offsets[None, :] * stride_in1_3
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load in_3[k, n]
        b_ptrs = in3_ptr + k_offsets[:, None] * stride_in3_0 + n_offsets[None, :] * stride_in3_1
        bb = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, bb)

    # Cast and store out[b, h, m, n]
    if DTYPE_CODE == 0:
        out_vals = acc.to(tl.float16)
    else:
        out_vals = acc.to(tl.bfloat16)

    out_ptrs = out_ptr + b * stride_out_0 + h * stride_out_1 + m_offsets[:, None] * stride_out_2 + n_offsets[None, :] * stride_out_3
    tl.store(out_ptrs, out_vals, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def flash_attn_kernel(
    matmul_ptr, stride_matmul_0, stride_matmul_1, stride_matmul_2, stride_matmul_3,
    in2_ptr, stride_in2_0, stride_in2_1, stride_in2_2, stride_in2_3, stride_in2_4,
    in0_ptr, stride_in0_0, stride_in0_1, stride_in0_2,
    in4_ptr, stride_in4_0, stride_in4_1, stride_in4_2,
    out_ptr, stride_out_0, stride_out_1, stride_out_2,
    B: tl.constexpr, N: tl.constexpr, D: tl.constexpr, HW: tl.constexpr,
    L: tl.constexpr, L_PAD: tl.constexpr, SLICE_START: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    DTYPE_CODE: tl.constexpr,
):
    """Flash attention kernel with inline relative position bias computation for BotNet."""
    batch_id = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    d_start = pid_d * BLOCK_D

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    d_offsets = d_start + tl.arange(0, BLOCK_D)

    m_mask = m_offsets < N
    d_mask = d_offsets < D

    # Initialize online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over key blocks
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # Compute spatial position components
        # ih = i % HW (query height), iw = i // HW (query width)
        # jh = j % HW (key height), jw = j // HW (key width)
        ih = m_offsets % HW   # [BLOCK_M]
        iw = m_offsets // HW  # [BLOCK_M]
        jh = n_offsets % HW   # [BLOCK_N]
        jw = n_offsets // HW  # [BLOCK_N]

        # Valid (i, j) pair mask
        ij_mask = m_mask[:, None] & n_mask[None, :]  # [BLOCK_M, BLOCK_N]

        # 1. Load in_0[b, i, j] - attention bias
        in0_ptrs = in0_ptr + batch_id * stride_in0_0 + m_offsets[:, None] * stride_in0_1 + n_offsets[None, :] * stride_in0_2
        in0 = tl.load(in0_ptrs, mask=ij_mask, other=0.0).to(tl.float32)

        # 2. Compute relative position bias from matmul result
        # idx = iw * L + jw + SLICE_START
        # matmul_w = idx // L_PAD, matmul_d = idx % L_PAD
        # value = matmul[b, ih, matmul_w, matmul_d] if matmul_d < L, else 0
        idx = iw[:, None] * L + jw[None, :] + SLICE_START  # [BLOCK_M, BLOCK_N]
        matmul_w = idx // L_PAD   # [BLOCK_M, BLOCK_N]
        matmul_d = idx % L_PAD    # [BLOCK_M, BLOCK_N]
        matmul_valid = matmul_d < L  # [BLOCK_M, BLOCK_N]

        # Load matmul[b, ih, matmul_w, matmul_d]
        matmul_ptrs = matmul_ptr + batch_id * stride_matmul_0 + ih[:, None] * stride_matmul_1 + matmul_w * stride_matmul_2 + matmul_d * stride_matmul_3
        matmul_val = tl.load(matmul_ptrs, mask=ij_mask & matmul_valid, other=0.0).to(tl.float32)

        # 3. Load in_2[b, iw, ih, jw, jh]
        # Compute offset: iw*stride1 + ih*stride2 for i, jw*stride3 + jh*stride4 for j
        in2_offset_i = iw * stride_in2_1 + ih * stride_in2_2  # [BLOCK_M]
        in2_offset_j = jw * stride_in2_3 + jh * stride_in2_4  # [BLOCK_N]
        in2_ptrs = in2_ptr + batch_id * stride_in2_0 + in2_offset_i[:, None] + in2_offset_j[None, :]
        in2 = tl.load(in2_ptrs, mask=ij_mask, other=0.0).to(tl.float32)

        # Compute total attention scores
        scores = in0 + matmul_val + in2  # [BLOCK_M, BLOCK_N] in fp32

        # Set out-of-bounds scores to -inf for correct softmax
        scores = tl.where(ij_mask, scores, float('-inf'))

        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))  # [BLOCK_M]
        # Safe correction factor: 0 when m_i is -inf (starting fresh)
        alpha = tl.where(m_i == float('-inf'), 0.0, tl.exp(m_i - m_i_new))  # [BLOCK_M]
        p = tl.exp(scores - m_i_new[:, None])  # [BLOCK_M, BLOCK_N]
        # Zero out p for out-of-bounds positions
        p = tl.where(ij_mask, p, 0.0)
        l_i_new = alpha * l_i + tl.sum(p, axis=1)  # [BLOCK_M]

        # Load values in_4[b, j, d] -> [BLOCK_N, BLOCK_D]
        in4_ptrs = in4_ptr + batch_id * stride_in4_0 + n_offsets[:, None] * stride_in4_1 + d_offsets[None, :] * stride_in4_2
        in4 = tl.load(in4_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # Cast p to value dtype for dot product
        if DTYPE_CODE == 0:
            p_cast = p.to(tl.float16)
        else:
            p_cast = p.to(tl.bfloat16)

        # Compute dot product: p_cast [BLOCK_M, BLOCK_N] @ in4 [BLOCK_N, BLOCK_D]
        dot_result = tl.dot(p_cast, in4)  # [BLOCK_M, BLOCK_D] in fp32

        # Update accumulator
        acc = alpha[:, None] * acc + dot_result

        m_i = m_i_new
        l_i = l_i_new

    # Finalize: output = acc / l_i
    output = acc / l_i[:, None]  # [BLOCK_M, BLOCK_D]

    # Store output[b, d, m] (transposed: [B, D, N])
    # output has shape [BLOCK_M, BLOCK_D], need to store as [BLOCK_D, BLOCK_M] block
    out_ptrs = out_ptr + batch_id * stride_out_0 + d_offsets[:, None] * stride_out_1 + m_offsets[None, :] * stride_out_2

    if DTYPE_CODE == 0:
        output_store = output.T.to(tl.float16)
    else:
        output_store = output.T.to(tl.bfloat16)

    tl.store(out_ptrs, output_store, mask=d_mask[:, None] & m_mask[None, :])


def _botnet_attn_impl(in_0, in_1, in_2, in_3, in_4, HW, L, N):
    """Implementation of BotNet attention for given spatial dimensions."""
    B = in_0.shape[0]
    D = in_4.shape[2]
    D_INNER = in_1.shape[3]
    L_PAD = L + 1
    SLICE_START = L - HW
    dtype = in_0.dtype
    device = in_0.device
    dtype_code = 0 if dtype == torch.float16 else 1

    # Step 1: Compute matmul1 = in_1 @ in_3
    matmul_result = torch.empty(B, HW, HW, L, dtype=dtype, device=device)

    BLOCK_M_MM = 16
    BLOCK_N_MM = 16
    BLOCK_K_MM = 32

    grid1 = (
        (HW + BLOCK_M_MM - 1) // BLOCK_M_MM,
        (L + BLOCK_N_MM - 1) // BLOCK_N_MM,
        B * HW
    )

    matmul1_kernel[grid1](
        in1_ptr=in_1, stride_in1_0=in_1.stride(0), stride_in1_1=in_1.stride(1),
        stride_in1_2=in_1.stride(2), stride_in1_3=in_1.stride(3),
        in3_ptr=in_3, stride_in3_0=in_3.stride(0), stride_in3_1=in_3.stride(1),
        out_ptr=matmul_result, stride_out_0=matmul_result.stride(0),
        stride_out_1=matmul_result.stride(1), stride_out_2=matmul_result.stride(2),
        stride_out_3=matmul_result.stride(3),
        B=B, HW=HW, D_INNER=D_INNER, L=L,
        BLOCK_M=BLOCK_M_MM, BLOCK_N=BLOCK_N_MM, BLOCK_K=BLOCK_K_MM,
        DTYPE_CODE=dtype_code,
    )

    # Step 2: Flash attention with inline bias computation
    output = torch.empty(B, D, N, dtype=dtype, device=device)

    BLOCK_M_ATTN = 16
    BLOCK_N_ATTN = 32
    BLOCK_D_ATTN = 32

    grid2 = (
        (N + BLOCK_M_ATTN - 1) // BLOCK_M_ATTN,
        (D + BLOCK_D_ATTN - 1) // BLOCK_D_ATTN,
        B
    )

    flash_attn_kernel[grid2](
        matmul_ptr=matmul_result, stride_matmul_0=matmul_result.stride(0),
        stride_matmul_1=matmul_result.stride(1), stride_matmul_2=matmul_result.stride(2),
        stride_matmul_3=matmul_result.stride(3),
        in2_ptr=in_2, stride_in2_0=in_2.stride(0), stride_in2_1=in_2.stride(1),
        stride_in2_2=in_2.stride(2), stride_in2_3=in_2.stride(3), stride_in2_4=in_2.stride(4),
        in0_ptr=in_0, stride_in0_0=in_0.stride(0), stride_in0_1=in_0.stride(1),
        stride_in0_2=in_0.stride(2),
        in4_ptr=in_4, stride_in4_0=in_4.stride(0), stride_in4_1=in_4.stride(1),
        stride_in4_2=in_4.stride(2),
        out_ptr=output, stride_out_0=output.stride(0), stride_out_1=output.stride(1),
        stride_out_2=output.stride(2),
        N=N, D=D, HW=HW, L=L, L_PAD=L_PAD, SLICE_START=SLICE_START,
        BLOCK_M=BLOCK_M_ATTN, BLOCK_N=BLOCK_N_ATTN, BLOCK_D=BLOCK_D_ATTN,
        DTYPE_CODE=dtype_code,
    )

    return (output,)


def _botnet_attn_256(in_0, in_1, in_2, in_3, in_4):
    """BotNet attention for 256x256 version (HW=16, L=31, N=256)."""
    return _botnet_attn_impl(in_0, in_1, in_2, in_3, in_4, HW=16, L=31, N=256)


def _botnet_attn_64(in_0, in_1, in_2, in_3, in_4):
    """BotNet attention for 64x64 version (HW=8, L=15, N=64)."""
    return _botnet_attn_impl(in_0, in_1, in_2, in_3, in_4, HW=8, L=15, N=64)


@torch.fx.wrap
def botnet_attn_dispatch(in_0, in_1, in_2, in_3, in_4, route):
    """Dispatch wrapper that routes to the appropriate implementation based on the route string."""
    if route == "256":
        return _botnet_attn_256(in_0, in_1, in_2, in_3, in_4)
    elif route == "64":
        return _botnet_attn_64(in_0, in_1, in_2, in_3, in_4)
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return botnet_attn_dispatch