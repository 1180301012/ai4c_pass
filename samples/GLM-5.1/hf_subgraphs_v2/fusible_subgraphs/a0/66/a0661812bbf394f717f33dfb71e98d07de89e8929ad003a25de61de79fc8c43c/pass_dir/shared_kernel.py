import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr,
    out_normal_ptr, out_transposed_ptr,
    M, N, K,
    S,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_on_m, stride_on_n,
    stride_ot_d, stride_ot_s,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Inner loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load input tile: (BLOCK_M, BLOCK_K) from (M, K) matrix
        input_ptrs = input_ptr + m_offsets[:, None] * stride_im + k_offsets[None, :] * stride_ik
        mask_mk = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        x = tl.load(input_ptrs, mask=mask_mk, other=0.0)

        # Load weight tile: weight is (N, K), load transposed as (BLOCK_K, BLOCK_N)
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        k_offsets2 = k_start + tl.arange(0, BLOCK_K)
        weight_ptrs = weight_ptr + k_offsets2[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        mask_kn = (k_offsets2[:, None] < K) & (n_offsets[None, :] < N)
        w = tl.load(weight_ptrs, mask=mask_kn, other=0.0)

        # Matrix multiply and accumulate
        acc += tl.dot(x, w, allow_tf32=True)

    # Add bias
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    b_ptrs = bias_ptr + n_offsets
    mask_n = n_offsets < N
    bias = tl.load(b_ptrs, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Cast to output dtype for stores
    if OUTPUT_DTYPE == "float16":
        acc_out = acc.to(tl.float16)
    elif OUTPUT_DTYPE == "bfloat16":
        acc_out = acc.to(tl.bfloat16)
    else:
        acc_out = acc  # float32, no cast needed

    # Store normal output: (BLOCK_M, BLOCK_N) tile at (m_start, n_start)
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    out_normal_ptrs = out_normal_ptr + m_offsets[:, None] * stride_on_m + n_offsets[None, :] * stride_on_n
    mask_mn = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(out_normal_ptrs, acc_out, mask=mask_mn)

    # Store transposed output: transpose the block and write to (N, M) layout
    # For B=1: transposed output is (D, S) = (N, M) with stride_d = S, stride_s = 1
    # acc_t has shape (BLOCK_N, BLOCK_M) matching (n_offsets, m_offsets) indexing
    acc_t = tl.trans(acc_out)
    out_transposed_ptrs = out_transposed_ptr + n_offsets[:, None] * stride_ot_d + m_offsets[None, :] * stride_ot_s
    mask_nm = (n_offsets[:, None] < N) & (m_offsets[None, :] < M)
    tl.store(out_transposed_ptrs, acc_t, mask=mask_nm)


@torch.fx.wrap
def fused_linear_transpose_dispatch(input_tensor, weight_tensor, bias_tensor, route):
    B = input_tensor.shape[0]
    S = input_tensor.shape[1]
    K = input_tensor.shape[2]
    D = weight_tensor.shape[0]
    M = B * S

    # Determine output dtype string for the kernel constexpr
    dtype = input_tensor.dtype
    if dtype == torch.float16:
        output_dtype_str = "float16"
    elif dtype == torch.bfloat16:
        output_dtype_str = "bfloat16"
    else:
        output_dtype_str = "float32"

    # Allocate output tensors
    out_normal = torch.empty((B, S, D), dtype=dtype, device=input_tensor.device)
    out_transposed = torch.empty((B, D, S), dtype=dtype, device=input_tensor.device)

    # Compute strides (assumes contiguous tensors)
    stride_im = input_tensor.stride(1)   # K for contiguous (B, S, K)
    stride_ik = input_tensor.stride(2)   # 1 for contiguous
    stride_wn = weight_tensor.stride(0)  # K for contiguous (D, K)
    stride_wk = weight_tensor.stride(1)  # 1 for contiguous
    stride_on_m = out_normal.stride(1)   # D for contiguous (B, S, D)
    stride_on_n = out_normal.stride(2)   # 1 for contiguous
    stride_ot_d = out_transposed.stride(1)  # S for contiguous (B, D, S)
    stride_ot_s = out_transposed.stride(2)  # 1 for contiguous

    # Grid: adaptive based on autotune config
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(D, META['BLOCK_N']),
    )

    fused_linear_transpose_kernel[grid](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        out_normal_ptr=out_normal,
        out_transposed_ptr=out_transposed,
        M=M, N=D, K=K,
        S=S,
        stride_im=stride_im, stride_ik=stride_ik,
        stride_wn=stride_wn, stride_wk=stride_wk,
        stride_on_m=stride_on_m, stride_on_n=stride_on_n,
        stride_ot_d=stride_ot_d, stride_ot_s=stride_ot_s,
        OUTPUT_DTYPE=output_dtype_str,
    )

    if route == "normal_transposed":
        return (out_normal, out_transposed)
    elif route == "transposed_normal":
        return (out_transposed, out_normal)
    else:
        raise ValueError(f"Unknown route: {route}")