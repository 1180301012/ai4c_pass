import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused linear kernel: output = input @ weight^T + bias
    
    Computes the linear operation in a single kernel, avoiding intermediate
    memory writes from dropout/cast operations that are identity.
    
    The input can be multi-dimensional (e.g., [B, S, K]) but is accessed
    as if flattened to [M, K] using computed strides.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs_base = tl.arange(0, BLOCK_K)

    # Accumulator in float32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_start * BLOCK_K + k_offs_base

        # Load input tile: a[m, k] = input[m, k] (using flattened strides), shape [BLOCK_M, BLOCK_K]
        a_ptrs = input_ptr + m_offs[:, None] * stride_im + k_offs[None, :] * stride_ik
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load weight tile (transposed): b[k, n] = weight[n, k], shape [BLOCK_K, BLOCK_N]
        # This gives us weight^T which is needed for linear: input @ weight^T
        b_ptrs = weight_ptr + n_offs[None, :] * stride_wn + k_offs[:, None] * stride_wk
        b_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply with accumulation: acc += a @ b
        acc = tl.dot(a, b, acc)

    # Add bias: output[m, n] = acc[m, n] + bias[n]
    bias_ptrs = bias_ptr + n_offs
    bias_mask = n_offs < N
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store output (using flattened strides)
    output_ptrs = output_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    output_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(output_ptrs, acc, mask=output_mask)


@torch.fx.wrap
def fused_linear(input, weight, bias):
    """Fused dropout(identity) + linear operation
    
    Since dropout with training=False is identity, we skip the dropout
    and directly compute the linear operation in a single Triton kernel.
    
    No reshape/view operations are used - strides are computed manually
    to handle multi-dimensional inputs as if flattened.
    """
    # Compute flattened dimensions
    # M = product of all leading dimensions, K = last dimension
    M = 1
    for i in range(input.dim() - 1):
        M = M * input.shape[i]
    K = input.shape[-1]
    N = weight.shape[0]

    # Allocate output with the correct multi-dimensional shape
    # Output shape: [*input_leading_dims, N]
    output_shape = list(input.shape[:-1]) + [N]
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Compute strides for "flattened" access
    # For a contiguous tensor [*leading, K]:
    #   - flattened stride for M dimension = K (stride of second-to-last dim)
    #   - flattened stride for K dimension = 1 (stride of last dim)
    stride_im = input.stride(-2) if input.dim() >= 2 else 1
    stride_ik = input.stride(-1)
    
    stride_wn = weight.stride(0)
    stride_wk = weight.stride(1)
    
    stride_om = output.stride(-2) if output.dim() >= 2 else 1
    stride_on = output.stride(-1)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    fused_linear_kernel[grid](
        input, weight, bias, output,
        M, N, K,
        stride_im, stride_ik,
        stride_wn, stride_wk,
        stride_om, stride_on,
    )

    return output