import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_view_transpose_kernel(
    hidden_ptr, weight_ptr, bias_ptr, out_ptr,
    B, S, H, D, K,  # K = H*D = hidden_dim
    stride_hb, stride_hs, stride_hk,
    stride_wn, stride_wk,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    """
    Fused kernel: linear(hidden, weight, bias) -> view(B, -1, H, D) -> transpose(1, 2)
    Computes: out[b, h, s, d] = sum_k(hidden[b, s, k] * weight[h*D+d, k]) + bias[h*D+d]
    Output is stored directly in [B, H, S, D] layout (transposed from natural [B, S, H*D]).
    """
    M = B * S  # total rows (one per (batch, seq) pair)
    N = H * D  # total cols (one per (head, dim) pair)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Decode row index: m -> (b, s)
    b_idx = offs_m // S
    s_idx = offs_m % S
    # Decode col index: n -> (h, d)
    h_idx = offs_n // D
    d_idx = offs_n % D

    # Accumulator in float32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load bias: bias[h*D+d] = bias[offs_n]
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]  # broadcast over BLOCK_M rows

    # Inner loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k

        # Load hidden[b, s, k]: shape [BLOCK_M, BLOCK_K]
        a_ptrs = hidden_ptr + b_idx[:, None] * stride_hb + s_idx[:, None] * stride_hs + k_offsets[None, :] * stride_hk
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load weight.T[k, n]: weight[n, k] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = weight_ptr + offs_n[None, :] * stride_wn + k_offsets[:, None] * stride_wk
        b_mask = (offs_n[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Matrix multiply: acc += a @ b (where b is weight.T)
        acc += tl.dot(a, b, allow_tf32=True)

    # Store output in [B, H, S, D] layout
    out_offsets = b_idx[:, None] * stride_ob + h_idx[None, :] * stride_oh + s_idx[:, None] * stride_os + d_idx[None, :] * stride_od
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


@triton.jit
def fused_linear_view_transpose_cast_kernel(
    hidden_ptr, weight_ptr, bias_ptr, out_ptr,
    B, S, H, D, K,
    stride_hb, stride_hs, stride_hk,
    stride_wn, stride_wk,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    CAST_DTYPE: tl.constexpr = 0,  # 0=no cast, 1=float16, 2=bfloat16
):
    """
    Fused kernel with optional dtype cast.
    Same as fused_linear_view_transpose_kernel but casts output to a different dtype.
    CAST_DTYPE: 0 = same as input, 1 = float16, 2 = bfloat16
    """
    M = B * S
    N = H * D

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    b_idx = offs_m // S
    s_idx = offs_m % S
    h_idx = offs_n // D
    d_idx = offs_n % D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k

        a_ptrs = hidden_ptr + b_idx[:, None] * stride_hb + s_idx[:, None] * stride_hs + k_offsets[None, :] * stride_hk
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        b_ptrs = weight_ptr + offs_n[None, :] * stride_wn + k_offsets[:, None] * stride_wk
        b_mask = (offs_n[None, :] < N) & (k_offsets[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b, allow_tf32=True)

    out_offsets = b_idx[:, None] * stride_ob + h_idx[None, :] * stride_oh + s_idx[:, None] * stride_os + d_idx[None, :] * stride_od
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Cast to target dtype before storing
    if CAST_DTYPE == 0:
        # No cast - store as-is (Triton handles dtype based on output tensor)
        tl.store(out_ptr + out_offsets, acc, mask=out_mask)
    elif CAST_DTYPE == 1:
        tl.store(out_ptr + out_offsets, acc.to(tl.float16), mask=out_mask)
    elif CAST_DTYPE == 2:
        tl.store(out_ptr + out_offsets, acc.to(tl.bfloat16), mask=out_mask)


def launch_fused_linear_view_transpose(bias, weight, hidden, H, D, out_dtype=None):
    """Launch the fused kernel with given parameters."""
    if out_dtype is None:
        out_dtype = hidden.dtype

    B = hidden.shape[0]
    S = hidden.shape[1]
    K = hidden.shape[2]  # hidden_dim = K
    N = H * D

    # Ensure weight and bias are on the same device as hidden
    device = hidden.device
    if weight.device != device:
        weight = weight.to(device)
    if bias.device != device:
        bias = bias.to(device)

    # Allocate output in [B, H, S, D] layout
    out = torch.empty((B, H, S, D), dtype=out_dtype, device=device)

    # Get strides
    stride_hb, stride_hs, stride_hk = hidden.stride()
    stride_wn, stride_wk = weight.stride()
    stride_ob, stride_oh, stride_os, stride_od = out.stride()

    # Determine grid and block sizes
    M = B * S
    # Choose block sizes based on problem size
    if M < 64:
        BLOCK_M = 16
    elif M < 256:
        BLOCK_M = 32
    else:
        BLOCK_M = 64

    if N < 64:
        BLOCK_N = 16
    elif N < 256:
        BLOCK_N = 32
    else:
        BLOCK_N = 64

    BLOCK_K = 32

    # Determine if we need a cast
    need_cast = (out_dtype != hidden.dtype)
    if need_cast:
        if out_dtype == torch.float16:
            CAST_DTYPE = 1
        elif out_dtype == torch.bfloat16:
            CAST_DTYPE = 2
        else:
            CAST_DTYPE = 0
    else:
        CAST_DTYPE = 0

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)

    if CAST_DTYPE == 0:
        fused_linear_view_transpose_kernel[grid](
            hidden_ptr=hidden, weight_ptr=weight, bias_ptr=bias, out_ptr=out,
            B=B, S=S, H=H, D=D, K=K,
            stride_hb=stride_hb, stride_hs=stride_hs, stride_hk=stride_hk,
            stride_wn=stride_wn, stride_wk=stride_wk,
            stride_ob=stride_ob, stride_oh=stride_oh, stride_os=stride_os, stride_od=stride_od,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    else:
        fused_linear_view_transpose_cast_kernel[grid](
            hidden_ptr=hidden, weight_ptr=weight, bias_ptr=bias, out_ptr=out,
            B=B, S=S, H=H, D=D, K=K,
            stride_hb=stride_hb, stride_hs=stride_hs, stride_hk=stride_hk,
            stride_wn=stride_wn, stride_wk=stride_wk,
            stride_ob=stride_ob, stride_oh=stride_oh, stride_os=stride_os, stride_od=stride_od,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            CAST_DTYPE=CAST_DTYPE,
        )

    return out


@torch.fx.wrap
def dispatch_fused_linear_view_transpose(bias, weight, hidden, route):
    """Dispatch wrapper that routes to the correct kernel based on route string."""
    # Route format: "B_H_D_nc" (no cast) or "B_H_D_cf16" (cast float16) or "B_H_D_cb16" (cast bfloat16)
    parts = route.split("_")
    B_view = int(parts[0])
    H = int(parts[1])
    D = int(parts[2])
    cast_type = parts[3] if len(parts) > 3 else "nc"

    if cast_type == "nc":
        out_dtype = hidden.dtype
    elif cast_type == "cf16":
        out_dtype = torch.float16
    elif cast_type == "cb16":
        out_dtype = torch.bfloat16
    elif cast_type == "cf32":
        out_dtype = torch.float32
    else:
        out_dtype = hidden.dtype

    return launch_fused_linear_view_transpose(bias, weight, hidden, H, D, out_dtype)