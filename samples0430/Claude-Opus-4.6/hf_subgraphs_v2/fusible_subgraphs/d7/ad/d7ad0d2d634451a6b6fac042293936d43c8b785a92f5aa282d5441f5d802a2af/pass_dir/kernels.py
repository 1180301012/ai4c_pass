import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    W_ptr, X_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_wm, stride_wk,
    stride_xb, stride_xn, stride_xk,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Computes: out[b, m, n] = sum_k W[m, k] * X[b, n, k] + bias[m]
    This effectively does: linear(X, W, bias) then permute(0,2,1)
    M=768 (output channels), N=256 (sequence length), K=512 (input channels)
    Output: [B, M, N] = [B, 768, 256] (transposed linear output)
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)

    # Swizzle for L2 cache locality
    num_pid_in_group = GROUP_M * num_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for W[M, K] and X_b^T[K, N] (accessed as X[b, n, k])
    a_ptrs = W_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    b_ptrs = X_ptr + pid_b * stride_xb + offs_k[:, None] * stride_xk + offs_n[None, :] * stride_xn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_rem = K - k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_wk
        b_ptrs += BLOCK_K * stride_xk

    # Add bias
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < M, other=0.0)
    acc += bias[:, None]

    # Store in transposed layout: out[b, m, n]
    c_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def bilinear_upsample_kernel(
    input_ptr, output_ptr,
    BC,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Upsamples from [BC, 16, 16] to [BC, 128, 128] using bilinear interpolation.
    align_corners=False, scale_factor=8x.
    """
    bc_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < 16384  # 128 * 128

    oy = offs // 128
    ox = offs % 128

    # Source coordinates (align_corners=False): src = (dst + 0.5) * (src_size/dst_size) - 0.5
    src_y = (oy.to(tl.float32) + 0.5) * 0.125 - 0.5
    src_x = (ox.to(tl.float32) + 0.5) * 0.125 - 0.5

    # Floor for bilinear indices
    y0 = tl.math.floor(src_y).to(tl.int32)
    x0 = tl.math.floor(src_x).to(tl.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    # Interpolation weights
    wy1 = src_y - y0.to(tl.float32)
    wx1 = src_x - x0.to(tl.float32)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    # Clamp coordinates to [0, 15]
    y0c = tl.maximum(tl.minimum(y0, 15), 0)
    y1c = tl.maximum(tl.minimum(y1, 15), 0)
    x0c = tl.maximum(tl.minimum(x0, 15), 0)
    x1c = tl.maximum(tl.minimum(x1, 15), 0)

    # Load 4 neighbors
    in_base = bc_idx * 256  # 16 * 16
    v00 = tl.load(input_ptr + in_base + y0c * 16 + x0c, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(input_ptr + in_base + y0c * 16 + x1c, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(input_ptr + in_base + y1c * 16 + x0c, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(input_ptr + in_base + y1c * 16 + x1c, mask=mask, other=0.0).to(tl.float32)

    # Bilinear interpolation
    result = wy0 * wx0 * v00 + wy0 * wx1 * v01 + wy1 * wx0 * v10 + wy1 * wx1 * v11

    # Store
    out_base = bc_idx * 16384  # 128 * 128
    tl.store(output_ptr + out_base + offs, result, mask=mask)


@torch.fx.wrap
def fused_linear_interpolate(bias, weight, x):
    """
    Fused: linear(x, weight, bias) -> permute(0,2,1) -> reshape(B,-1,16,16) -> bilinear_interpolate(128,128)
    """
    B = x.shape[0]
    M_seq = x.shape[1]  # 256
    K = x.shape[2]  # 512
    C = weight.shape[0]  # 768

    device = x.device
    dtype = x.dtype

    # Intermediate: [B, C, M_seq] = [B, 768, 256]
    # This is the transposed linear output (fuses permute+reshape)
    intermediate = torch.empty((B, C, M_seq), dtype=dtype, device=device)

    # Matmul with transposed output
    M_mm = C  # 768
    N_mm = M_seq  # 256
    K_mm = K  # 512

    grid_mm = lambda META: (
        triton.cdiv(M_mm, META['BLOCK_M']) * triton.cdiv(N_mm, META['BLOCK_N']),
        B
    )

    matmul_bias_kernel[grid_mm](
        weight, x, bias, intermediate,
        M_mm, N_mm, K_mm,
        weight.stride(0), weight.stride(1),
        x.stride(0), x.stride(1), x.stride(2),
        intermediate.stride(0), intermediate.stride(1), intermediate.stride(2),
    )

    # Bilinear upsample: [B*C, 16, 16] -> [B*C, 128, 128]
    BC = B * C
    output = torch.empty((B, C, 128, 128), dtype=dtype, device=device)

    BLOCK_SIZE = 1024
    num_blocks = 16  # 128*128 / 1024 = 16

    bilinear_upsample_kernel[(BC, num_blocks)](
        intermediate, output,
        BC,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output