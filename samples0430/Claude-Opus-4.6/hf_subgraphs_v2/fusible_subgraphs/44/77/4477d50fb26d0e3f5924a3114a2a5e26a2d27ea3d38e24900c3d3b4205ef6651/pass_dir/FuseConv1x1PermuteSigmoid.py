import torch
import triton
import triton.language as tl


def pattern(bias, weight, x, batch_size, last_dim):
    conv2d = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(batch_size, -1, last_dim)
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return tmp_5


def replacement_args(bias, weight, x, batch_size, last_dim):
    return (bias, weight, x, batch_size, last_dim)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def conv1x1_sigmoid_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,  # total spatial positions = N_batch * H * W
    K,  # input channels = C_in
    N,  # output channels = C_out
    HW,  # H * W
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)

    # Compute batch and spatial indices
    n_batch_idx = m_offsets // HW
    hw_idx = m_offsets % HW

    # Accumulator for matmul result
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over input channels
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load input tile [BLOCK_M, BLOCK_K]
        # input layout: [N_batch, C_in, H, W] (NCHW)
        # input[n, k, hw] at offset: n * K * HW + k * HW + hw
        a_ptrs = n_batch_idx[:, None] * (K * HW) + k_offsets[None, :] * HW + hw_idx[:, None]
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(input_ptr + a_ptrs, mask=a_mask, other=0.0)

        # Load weight^T tile [BLOCK_K, BLOCK_N]
        # weight layout: [C_out, C_in] (reshaped from [C_out, C_in, 1, 1])
        # weight^T[k, n] = weight[n, k] at offset: n * K + k
        b_ptrs = n_offsets[None, :] * K + k_offsets[:, None]
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(weight_ptr + b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply and accumulate
        acc += tl.dot(a, b)

    # Add bias [BLOCK_N]
    bias_val = tl.load(bias_ptr + n_offsets, mask=n_offsets < N, other=0.0)
    acc += bias_val[None, :]

    # Apply sigmoid activation
    acc = tl.sigmoid(acc)

    # Store output in [M, N] layout
    out_ptrs = m_offsets[:, None] * N + n_offsets[None, :]
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(output_ptr + out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_sigmoid(bias, weight, x, batch_size, last_dim):
    # Extract dimensions
    N_batch = x.shape[0]
    C_in = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    C_out = weight.shape[0]

    M = N_batch * H * W
    HW = H * W

    # Compute BLOCK_N: power of 2, >= 16 (for tl.dot compatibility)
    BLOCK_N = 16
    while BLOCK_N < C_out:
        BLOCK_N *= 2

    # Allocate output tensor in final shape
    output = torch.empty((batch_size, M // batch_size, last_dim), dtype=x.dtype, device=x.device)

    # Launch kernel
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    conv1x1_sigmoid_kernel[grid](
        input_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        M=M,
        K=C_in,
        N=C_out,
        HW=HW,
        BLOCK_N=BLOCK_N,
    )

    return output


def replacement_func():
    return fused_conv1x1_sigmoid