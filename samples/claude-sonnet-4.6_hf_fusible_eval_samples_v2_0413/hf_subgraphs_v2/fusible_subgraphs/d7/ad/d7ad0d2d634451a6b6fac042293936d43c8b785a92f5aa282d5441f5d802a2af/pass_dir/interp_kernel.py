import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton GEMM + bias kernel: computes (M, K) @ (N, K).T + (N,) -> (M, N)
# Input x is viewed as (B*SEQ, K), weight is (N, K), bias is (N,).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_kernel(
    a_ptr,   # (M, K) input x flattened
    b_ptr,   # (N, K) weight
    bias_ptr, # (N,) bias
    c_ptr,   # (M, N) output
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # b stored as (N, K): stride along K-dim = 1, N-dim = K
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + n_offs[:, None] * stride_bk + k_offs[None, :] * stride_bn

        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, tl.trans(b))

    # Add bias
    bias = tl.load(bias_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store output
    c_ptrs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)

    # Load a single element to get dtype
    sample = tl.load(a_ptr, mask=True, other=0.0)
    tl.store(c_ptrs, acc.to(sample.dtype), mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=2),
    ],
    key=['B', 'C'],
)
@triton.jit
def _fused_perm_interp_kernel(
    input_ptr,   # [B, 256, C]  (linear output, with seq=256=16*16)
    output_ptr,  # [B, C, 128, 128]
    B, C,
    BLOCK_HW: tl.constexpr,
):
    # Fixed constants: source 16x16, output 128x128, scale=1/8
    SEQ: tl.constexpr = 256
    W_IN: tl.constexpr = 16
    H_IN: tl.constexpr = 16
    W_OUT: tl.constexpr = 128
    H_OUT: tl.constexpr = 128
    HW_OUT: tl.constexpr = 16384  # 128*128

    pid_bc = tl.program_id(0)   # flattened (b, c) index
    pid_hw = tl.program_id(1)   # spatial tile index

    b = pid_bc // C
    c = pid_bc % C

    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < HW_OUT

    h_out = hw_offsets // W_OUT
    w_out = hw_offsets % W_OUT

    # align_corners=False bilinear: h_src = (h_out+0.5)*(H_IN/H_OUT) - 0.5
    h_src = (h_out.to(tl.float32) + 0.5) * 0.125 - 0.5
    w_src = (w_out.to(tl.float32) + 0.5) * 0.125 - 0.5

    h0 = tl.floor(h_src).to(tl.int32)
    w0 = tl.floor(w_src).to(tl.int32)
    h1 = h0 + 1
    w1 = w0 + 1

    dh = h_src - h0.to(tl.float32)
    dw = w_src - w0.to(tl.float32)

    # Clamp to valid source range [0, H_IN-1]
    h0c = tl.maximum(0, tl.minimum(H_IN - 1, h0))
    h1c = tl.maximum(0, tl.minimum(H_IN - 1, h1))
    w0c = tl.maximum(0, tl.minimum(W_IN - 1, w0))
    w1c = tl.maximum(0, tl.minimum(W_IN - 1, w1))

    # In source [B, SEQ, C], element at [b, h*W_IN+w, c]
    seq00 = h0c * W_IN + w0c
    seq01 = h0c * W_IN + w1c
    seq10 = h1c * W_IN + w0c
    seq11 = h1c * W_IN + w1c

    # Base address for this (b, c): input[b, :, c] starts at b*SEQ*C + c
    base = b * SEQ * C + c

    v00 = tl.load(input_ptr + base + seq00 * C, mask=hw_mask, other=0.0)
    v01 = tl.load(input_ptr + base + seq01 * C, mask=hw_mask, other=0.0)
    v10 = tl.load(input_ptr + base + seq10 * C, mask=hw_mask, other=0.0)
    v11 = tl.load(input_ptr + base + seq11 * C, mask=hw_mask, other=0.0)

    # Upcast to float32 for computation
    v00f = v00.to(tl.float32)
    v01f = v01.to(tl.float32)
    v10f = v10.to(tl.float32)
    v11f = v11.to(tl.float32)

    # Bilinear weights
    w00 = (1.0 - dh) * (1.0 - dw)
    w01 = (1.0 - dh) * dw
    w10 = dh * (1.0 - dw)
    w11 = dh * dw

    result = w00 * v00f + w01 * v01f + w10 * v10f + w11 * v11f

    # Cast result back to input dtype and store
    result_typed = result.to(v00.dtype)

    # Output layout: [B, C, H_OUT, W_OUT] contiguous
    out_idx = b * (C * HW_OUT) + c * HW_OUT + h_out * W_OUT + w_out
    tl.store(output_ptr + out_idx, result_typed, mask=hw_mask)


@torch.fx.wrap
def fused_perm_interp(x):
    """
    Fused permute(0,2,1) + reshape(B,-1,16,16) + bilinear_interpolate(128,128).
    Input x: [B, 256, C]  (output of linear layer)
    Output:  [B, C, 128, 128]
    """
    B = x.shape[0]
    C = x.shape[2]   # x is [B, 256, C]
    H_OUT = 128
    W_OUT = 128
    HW_OUT = H_OUT * W_OUT  # 16384

    out = torch.empty((B, C, H_OUT, W_OUT), dtype=x.dtype, device=x.device)

    def grid(meta):
        return (B * C, triton.cdiv(HW_OUT, meta['BLOCK_HW']))

    _fused_perm_interp_kernel[grid](x, out, B, C)

    return out


# ---------------------------------------------------------------------------
# Copy-transpose kernel: [B*SEQ, N] → [B, N, SEQ]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SEQ': 64}, num_warps=4),
        triton.Config({'BLOCK_SEQ': 128}, num_warps=4),
        triton.Config({'BLOCK_SEQ': 256}, num_warps=8),
    ],
    key=['B', 'N'],
)
@triton.jit
def _copy_perm_kernel(
    src_ptr,  # [M, N]  = [B*SEQ, N] row-major
    dst_ptr,  # [B, N, SEQ] contiguous
    B, N,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    pid_bn  = tl.program_id(0)   # b * N + n
    pid_seq = tl.program_id(1)

    b = pid_bn // N
    n = pid_bn % N

    seq_start = pid_seq * BLOCK_SEQ
    seq_offs  = seq_start + tl.arange(0, BLOCK_SEQ)
    mask      = seq_offs < SEQ

    # src[b*SEQ + seq, n] = src_ptr[(b*SEQ + seq)*N + n]
    src_base = b * SEQ * N + n
    vals = tl.load(src_ptr + src_base + seq_offs * N, mask=mask, other=0.0)

    # dst[b, n, seq] = dst_ptr[b*N*SEQ + n*SEQ + seq]
    dst_base = b * N * SEQ + n * SEQ
    tl.store(dst_ptr + dst_base + seq_offs, vals, mask=mask)


@torch.fx.wrap
def fused_linear_view(in_0, in_1, in_2):
    """
    Fused linear(in_2, in_1, in_0) + permute(0,2,1) + reshape(B,-1,16,16)
    Returns [B, N, 128, 128] via Triton GEMM + bilinear upsampling,
    so downstream F.interpolate(size=(128,128)) becomes an identity pass-through.
    in_0: bias   [N]      e.g. [768]
    in_1: weight [N, K]   e.g. [768, 512]
    in_2: x      [B,256,K]
    """
    B   = in_2.shape[0]
    SEQ = in_2.shape[1]   # 256
    K   = in_2.shape[2]   # 512
    N   = in_1.shape[0]   # 768

    device = in_2.device
    dtype  = in_2.dtype

    # Move weight/bias to the correct device+dtype (they may be on CPU)
    bias_d   = in_0.to(device=device, dtype=dtype)
    weight_d = in_1.to(device=device, dtype=dtype)
    x_cont   = in_2.contiguous()

    M = B * SEQ
    # Allocate output with torch.empty (allowed)
    linear_flat = torch.empty((M, N), dtype=dtype, device=device)

    def gemm_grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _gemm_bias_kernel[gemm_grid](
        x_cont, weight_d, bias_d, linear_flat,
        M, N, K,
        K, 1,   # stride_am, stride_ak
        K, 1,   # stride_bk, stride_bn  (weight [N,K] row-major)
        N, 1,   # stride_cm, stride_cn
    )

    # linear_flat is freshly-allocated → .view() is allowed
    linear_3d = linear_flat.view(B, SEQ, N)   # [B, SEQ, N]

    # Fused permute(0,2,1) + reshape + bilinear 16x16→128x128
    # Returns [B, N, 128, 128]; downstream F.interpolate(128→128) is identity
    return fused_perm_interp(linear_3d)


@torch.fx.wrap
def fused_linear_perm_interp(bias, weight, x):
    """
    Full fused: linear(x, weight, bias) + permute(0,2,1) + reshape(B,-1,16,16)
                + bilinear_interpolate(128,128, align_corners=False).
    bias:   [N]       e.g. [768]
    weight: [N, K]    e.g. [768, 512]
    x:      [B, SEQ, K]  e.g. [B, 256, 512]
    Output: [B, N, 128, 128]
    """
    B = x.shape[0]
    SEQ = x.shape[1]   # 256
    K = x.shape[2]     # 512
    N = weight.shape[0]  # 768

    device = x.device
    dtype = x.dtype

    # Move weight/bias to the same device+dtype as input (they may start on CPU)
    bias_d = bias.to(device=device, dtype=dtype)
    weight_d = weight.to(device=device, dtype=dtype)
    x_cont = x.contiguous()

    M = B * SEQ
    linear_out_flat = torch.empty((M, N), dtype=dtype, device=device)

    def gemm_grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _gemm_bias_kernel[gemm_grid](
        x_cont, weight_d, bias_d, linear_out_flat,
        M, N, K,
        K, 1,   # stride_am, stride_ak  (A is [M,K] row-major)
        K, 1,   # stride_bk, stride_bn  (weight is [N,K] row-major)
        N, 1,   # stride_cm, stride_cn  (C is [M,N] row-major)
    )

    # Reshape to [B, SEQ, N] then fuse permute+reshape+interp
    linear_out_3d = linear_out_flat.view(B, SEQ, N)
    result = fused_perm_interp(linear_out_3d)
    return (result,)