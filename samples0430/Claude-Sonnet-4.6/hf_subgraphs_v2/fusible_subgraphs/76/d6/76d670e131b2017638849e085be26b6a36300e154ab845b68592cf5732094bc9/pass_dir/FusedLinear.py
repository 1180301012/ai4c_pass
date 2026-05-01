import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Pattern: F.linear(x, weight, bias)  →  x @ weight.T + bias
# -----------------------------------------------------------------------

def pattern(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


# -----------------------------------------------------------------------
# Triton GEMM kernel
#   x      : [M, K]
#   weight : [N, K]   (stored row-major; each row is an output neuron)
#   bias   : [N]
#   out    : [M, N]
#
#   Computes: out = x @ weight.T + bias
#
#   OUTPUT_DTYPE: constexpr tl dtype so the store cast is known at compile time
# -----------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    # Grouped-ordering of programs for better L2 cache reuse
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Pointers for the first K-tile of x and w
    x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
    w_ptrs = w_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + k_offs) < K

        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # x : [BLOCK_M, BLOCK_K]
        # w : [BLOCK_N, BLOCK_K]  → transpose → [BLOCK_K, BLOCK_N]
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
    acc += b[None, :]

    # Store result using the compile-time OUTPUT_DTYPE
    out_ptrs = out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc.to(OUTPUT_DTYPE), mask=m_mask[:, None] & n_mask[None, :])


# Map torch dtypes to Triton dtypes
_TORCH_TO_TRITON_DTYPE = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}

# -----------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------

@torch.fx.wrap
def fused_linear(x, weight, bias):
    # x: [M, K],  weight: [N, K],  bias: [N]
    M = x.shape[0]
    K = x.shape[1]
    N = weight.shape[0]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    OUTPUT_DTYPE = _TORCH_TO_TRITON_DTYPE[x.dtype]

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        OUTPUT_DTYPE=OUTPUT_DTYPE,
    )
    return out


def replacement_func():
    return fused_linear