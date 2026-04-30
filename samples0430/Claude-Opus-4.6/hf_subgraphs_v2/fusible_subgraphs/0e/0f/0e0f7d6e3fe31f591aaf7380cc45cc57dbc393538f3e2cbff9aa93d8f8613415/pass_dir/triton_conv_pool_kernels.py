import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    H_out, W_out,
    C_in, H_in, W_in,
    K_h: tl.constexpr, K_w: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_oc, weight_stride_ic, weight_stride_kh, weight_stride_kw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Implicit GEMM conv2d kernel.
    M = batch * H_out * W_out (spatial positions)
    N = C_out (output channels)
    K = C_in * K_h * K_w (reduction dim)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decompose m into (batch_idx, h_out, w_out)
    hw_out = H_out * W_out
    batch_idx = m_offs // hw_out
    rem = m_offs % hw_out
    h_out_idx = rem // W_out
    w_out_idx = rem % W_out

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Decompose k into (c_in, kh, kw)
        kh_kw = K_h * K_w
        c_in_idx = k_offs // kh_kw
        k_rem = k_offs % kh_kw
        kh_idx = k_rem // K_w
        kw_idx = k_rem % K_w

        # Compute input spatial positions [BLOCK_M, BLOCK_K]
        h_in_pos = h_out_idx[:, None] * stride_h + kh_idx[None, :] - pad_h
        w_in_pos = w_out_idx[:, None] * stride_w + kw_idx[None, :] - pad_w

        # Masks
        valid_h = (h_in_pos >= 0) & (h_in_pos < H_in)
        valid_w = (w_in_pos >= 0) & (w_in_pos < W_in)
        valid_m = m_offs[:, None] < M
        valid_k = k_offs[None, :] < K
        a_mask = valid_h & valid_w & valid_m & valid_k

        # Load input [BLOCK_M, BLOCK_K]
        a_ptrs = (batch_idx[:, None] * input_stride_n +
                  c_in_idx[None, :] * input_stride_c +
                  h_in_pos * input_stride_h +
                  w_in_pos * input_stride_w)
        a = tl.load(input_ptr + a_ptrs, mask=a_mask, other=0.0)

        # Load weight [BLOCK_K, BLOCK_N]
        # weight layout: [C_out, C_in, K_h, K_w]
        w_ptrs = (n_offs[None, :] * weight_stride_oc +
                  c_in_idx[:, None] * weight_stride_ic +
                  kh_idx[:, None] * weight_stride_kh +
                  kw_idx[:, None] * weight_stride_kw)
        valid_n = n_offs[None, :] < N
        valid_k2 = k_offs[:, None] < K
        b_mask = valid_n & valid_k2
        b = tl.load(weight_ptr + w_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply
        acc += tl.dot(a, b)

    # Store output [BLOCK_M, BLOCK_N]
    # Output layout is contiguous [batch, C_out, H_out, W_out]
    out_ptrs = (batch_idx[:, None] * (N * hw_out) +
                n_offs[None, :] * hw_out +
                h_out_idx[:, None] * W_out +
                w_out_idx[:, None])
    out_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(output_ptr + out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['total_elements'],
)
@triton.jit
def maxpool2d_kernel(
    input_ptr, output_ptr,
    total_elements,
    C, H_in, W_in, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Max pool 2d with kernel=3, stride=2, padding=1.
    Input/output layout: [N, C, H, W] contiguous.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # Decompose linear index into (n, c, oh, ow)
    ow = offs % W_out
    tmp = offs // W_out
    oh = tmp % H_out
    tmp2 = tmp // H_out
    c = tmp2 % C
    n = tmp2 // C

    # Compute base input pointer for this (n, c)
    base = n * (C * H_in * W_in) + c * (H_in * W_in)

    # Max pool: kernel=3, stride=2, padding=1
    max_val = tl.full((BLOCK_SIZE,), value=float('-inf'), dtype=tl.float32)

    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            h_in = oh * 2 + kh - 1
            w_in = ow * 2 + kw - 1
            valid = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in) & mask
            in_idx = base + h_in * W_in + w_in
            val = tl.load(input_ptr + in_idx, mask=valid, other=float('-inf'))
            val_f32 = val.to(tl.float32)
            max_val = tl.maximum(max_val, val_f32)

    tl.store(output_ptr + offs, max_val.to(output_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_conv_pool(weight, input_tensor):
    """
    Fused conv2d + max_pool2d.
    Determines conv parameters from weight shape.
    Weight: [C_out, C_in, K_h, K_w]
    Input: [N, C_in, H_in, W_in]
    """
    # Get shapes
    C_out = weight.shape[0]
    C_in = weight.shape[1]
    K_h = weight.shape[2]
    K_w = weight.shape[3]
    batch = input_tensor.shape[0]
    H_in = input_tensor.shape[2]
    W_in = input_tensor.shape[3]

    # Determine conv parameters from kernel size
    if K_h == 7:
        stride_h, stride_w = 2, 2
        pad_h, pad_w = 3, 3
    else:
        stride_h, stride_w = 1, 1
        pad_h, pad_w = 1, 1

    # Conv output dimensions
    H_conv = (H_in + 2 * pad_h - K_h) // stride_h + 1
    W_conv = (W_in + 2 * pad_w - K_w) // stride_w + 1

    # Pool output dimensions (kernel=3, stride=2, padding=1)
    H_pool = (H_conv + 2 * 1 - 3) // 2 + 1
    W_pool = (W_conv + 2 * 1 - 3) // 2 + 1

    # GEMM dimensions
    M = batch * H_conv * W_conv
    N = C_out
    K = C_in * K_h * K_w

    # Make sure inputs are contiguous
    input_c = input_tensor.contiguous()
    weight_c = weight.contiguous()

    # Allocate intermediate (conv output)
    conv_out = torch.empty((batch, C_out, H_conv, W_conv),
                           dtype=input_tensor.dtype, device=input_tensor.device)

    # Launch conv2d kernel
    grid_conv = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    conv2d_implicit_gemm_kernel[grid_conv](
        input_c, weight_c, conv_out,
        M, N, K,
        H_conv, W_conv,
        C_in, H_in, W_in,
        K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        input_c.stride(0), input_c.stride(1), input_c.stride(2), input_c.stride(3),
        weight_c.stride(0), weight_c.stride(1), weight_c.stride(2), weight_c.stride(3),
    )

    # Allocate pool output
    pool_out = torch.empty((batch, C_out, H_pool, W_pool),
                           dtype=input_tensor.dtype, device=input_tensor.device)

    # Launch maxpool kernel
    total_pool = batch * C_out * H_pool * W_pool
    grid_pool = lambda meta: (triton.cdiv(total_pool, meta['BLOCK_SIZE']),)

    maxpool2d_kernel[grid_pool](
        conv_out, pool_out,
        total_pool,
        C_out, H_conv, W_conv, H_pool, W_pool,
    )

    return pool_out