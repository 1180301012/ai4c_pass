import torch
import triton
import triton.language as tl


def pattern(in_10, in_8, in_7):
    conv2d = torch.conv2d(in_10, in_8, in_7, (1, 1), (0, 0), (1, 1), 1)
    tmp_11 = torch.nn.functional.interpolate(conv2d, size = (512, 512), mode = 'bilinear', align_corners = False)
    return tmp_11


def replacement_args(in_10, in_8, in_7):
    return (in_10, in_8, in_7)


# ============================================================
# Triton Matmul Kernel for 1x1 Conv (Transposed Approach)
# Computes: C^T[n, m] = sum_k B^T[n, k] * A^T[k, m] + bias[n]
# Where B^T[n, k] = weight[n, k], A^T[k, m] = input[k, m]
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_ak, stride_am,
    stride_bn, stride_bk,
    stride_cn, stride_cm,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache optimization: group programs along M dimension
    GROUP_SIZE_M: tl.constexpr = 8
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    n_mask = n_offsets < N
    m_mask = m_offsets < M

    # Accumulator in float32
    acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load B^T tile: B^T[n, k] = weight[n, k]
        # Shape: [BLOCK_N, BLOCK_K]
        b = tl.load(
            b_ptr + n_offsets[:, None] * stride_bn + k_offsets[None, :] * stride_bk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Load A^T tile: A^T[k, m] = input[k, m]
        # Shape: [BLOCK_K, BLOCK_M]
        a = tl.load(
            a_ptr + k_offsets[:, None] * stride_ak + m_offsets[None, :] * stride_am,
            mask=k_mask[:, None] & m_mask[None, :],
            other=0.0,
        )

        # Compute dot product: acc += b @ a
        # b: [BLOCK_N, BLOCK_K], a: [BLOCK_K, BLOCK_M]
        # Result: [BLOCK_N, BLOCK_M]
        acc += tl.dot(b, a, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
    acc += bias[:, None]

    # Store C^T tile: C^T[n, m] = output[n, m]
    tl.store(
        c_ptr + n_offsets[:, None] * stride_cn + m_offsets[None, :] * stride_cm,
        acc,
        mask=n_mask[:, None] & m_mask[None, :],
    )


# ============================================================
# Triton Bilinear Interpolation Kernel (2D Tiled)
# Upsamples from [C, H_in, W_in] to [C, H_out, W_out]
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'TILE_H': 8, 'TILE_W': 32}, num_warps=4),
        triton.Config({'TILE_H': 16, 'TILE_W': 32}, num_warps=4),
        triton.Config({'TILE_H': 16, 'TILE_W': 64}, num_warps=4),
        triton.Config({'TILE_H': 32, 'TILE_W': 32}, num_warps=4),
        triton.Config({'TILE_H': 32, 'TILE_W': 64}, num_warps=8),
        triton.Config({'TILE_H': 64, 'TILE_W': 32}, num_warps=4),
        triton.Config({'TILE_H': 64, 'TILE_W': 64}, num_warps=8),
    ],
    key=['H_in', 'W_in', 'H_out', 'W_out', 'C'],
)
@triton.jit
def interpolate_bilinear_kernel(
    input_ptr, output_ptr,
    H_in, W_in, H_out, W_out, C,
    stride_ic, stride_ih, stride_iw,
    stride_oc, stride_oh, stride_ow,
    TILE_H: tl.constexpr, TILE_W: tl.constexpr,
):
    # 3D grid: (channel, row_block, col_block)
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    c = pid_c
    oy_base = pid_h * TILE_H
    ox_base = pid_w * TILE_W

    oy = oy_base + tl.arange(0, TILE_H)
    ox = ox_base + tl.arange(0, TILE_W)

    oy_mask = oy < H_out
    ox_mask = ox < W_out

    # Source coordinates (align_corners=False)
    scale_h = H_in / H_out
    scale_w = W_in / W_out

    sy = (oy.to(tl.float32) + 0.5) * scale_h - 0.5
    sx = (ox.to(tl.float32) + 0.5) * scale_w - 0.5

    # Integer coordinates (clipped to valid range)
    iy0 = tl.maximum(tl.floor(sy).to(tl.int32), 0)
    iy1 = tl.minimum(iy0 + 1, H_in - 1)
    ix0 = tl.maximum(tl.floor(sx).to(tl.int32), 0)
    ix1 = tl.minimum(ix0 + 1, W_in - 1)

    # Interpolation weights
    wy1 = sy - iy0.to(tl.float32)
    wx1 = sx - ix0.to(tl.float32)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    # Base offset for this channel in the input
    base = c * stride_ic

    # Load 4 corner values for bilinear interpolation
    # Cast to float32 for accurate computation
    mask_2d = oy_mask[:, None] & ox_mask[None, :]

    v00 = tl.load(
        input_ptr + base + iy0[:, None] * stride_ih + ix0[None, :] * stride_iw,
        mask=mask_2d, other=0.0,
    ).to(tl.float32)
    v01 = tl.load(
        input_ptr + base + iy0[:, None] * stride_ih + ix1[None, :] * stride_iw,
        mask=mask_2d, other=0.0,
    ).to(tl.float32)
    v10 = tl.load(
        input_ptr + base + iy1[:, None] * stride_ih + ix0[None, :] * stride_iw,
        mask=mask_2d, other=0.0,
    ).to(tl.float32)
    v11 = tl.load(
        input_ptr + base + iy1[:, None] * stride_ih + ix1[None, :] * stride_iw,
        mask=mask_2d, other=0.0,
    ).to(tl.float32)

    # Bilinear interpolation
    result = wy0[:, None] * (wx0[None, :] * v00 + wx1[None, :] * v01) + \
             wy1[:, None] * (wx0[None, :] * v10 + wx1[None, :] * v11)

    # Store output
    tl.store(
        output_ptr + c * stride_oc + oy[:, None] * stride_oh + ox[None, :] * stride_ow,
        result,
        mask=mask_2d,
    )


# ============================================================
# Wrapper Function
# ============================================================

@torch.fx.wrap
def fused_conv1x1_interpolate(input_tensor, weight, bias):
    C_in = weight.shape[1]
    C_out = weight.shape[0]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    M = H * W
    K = C_in
    N = C_out

    dtype = input_tensor.dtype
    device = input_tensor.device

    # Allocate conv output: [1, C_out, H, W]
    conv_out = torch.empty([1, C_out, H, W], dtype=dtype, device=device)

    # Strides for transposed matmul:
    # A^T[k, m] = input[k, m]: stride_ak = H*W, stride_am = 1
    # B^T[n, k] = weight[n, k]: stride_bn = C_in, stride_bk = 1
    # C^T[n, m] = output[n, m]: stride_cn = H*W, stride_cm = 1

    HW = H * W

    grid_matmul = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    conv1x1_matmul_kernel[grid_matmul](
        input_tensor, weight, conv_out, bias,
        M, N, K,
        HW, 1,        # stride_ak, stride_am
        C_in, 1,       # stride_bn, stride_bk
        HW, 1,         # stride_cn, stride_cm
        BLOCK_N=32, BLOCK_M=64, BLOCK_K=32,  # defaults, overridden by autotune
    )

    # Interpolate from [1, C_out, H, W] to [1, C_out, H_out, W_out]
    H_out = 512
    W_out = 512
    C = C_out

    output = torch.empty([1, C, H_out, W_out], dtype=dtype, device=device)

    grid_interp = lambda META: (
        C,
        triton.cdiv(H_out, META['TILE_H']),
        triton.cdiv(W_out, META['TILE_W']),
    )

    interpolate_bilinear_kernel[grid_interp](
        conv_out, output,
        H, W, H_out, W_out, C,
        HW, W, 1,                # stride_ic, stride_ih, stride_iw
        H_out * W_out, W_out, 1, # stride_oc, stride_oh, stride_ow
        TILE_H=16, TILE_W=32,    # defaults, overridden by autotune
    )

    return output


def replacement_func():
    return fused_conv1x1_interpolate