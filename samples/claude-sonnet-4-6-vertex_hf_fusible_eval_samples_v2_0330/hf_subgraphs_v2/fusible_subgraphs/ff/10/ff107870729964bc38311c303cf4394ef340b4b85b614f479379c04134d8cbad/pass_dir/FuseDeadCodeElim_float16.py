"""
Pass: FuseDeadCodeElim_float16

Matches the full float16 computation graph including the dead-code branch:
  conv2d → interpolate → tmp_11  (RETURNED)
  conv2d_1 → batch_norm → relu → .to(float16) → conv2d_2 → tmp_16  (DEAD, never returned)

Replaces with pure-Triton kernels:
  1. Eliminates all dead-code computation
  2. Triton GEMM for the 1x1 conv (weight[Cout,Cin] @ input.view(Cin,HW) + bias)
  3. Triton bilinear upsampling kernel
"""

import torch
import triton
import triton.language as tl


# ── GEMM kernel (1×1 conv as matmul): C[M,N] = A[M,K] @ B[K,N] + bias[M] ──

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_f16_dce_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """GEMM with bias: C = A @ B + bias[row].  Output stored as float16."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Row/col offsets, using modulo wrap to avoid out-of-bound loads
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k  = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk  + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias along the M (output-channel) axis
    offs_am_true = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    bias = tl.load(bias_ptr + offs_am_true, mask=offs_am_true < M, other=0.0).to(tl.float32)
    acc += bias[:, None]

    # Masked store
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs  = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask  = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# ── Bilinear upsampling kernel (align_corners=False), float16 output ──

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['N', 'C', 'H_in', 'W_in', 'H_out', 'W_out'],
)
@triton.jit
def bilinear_upsample_f16_dce_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    scale_h, scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total   = N * C * H_out * W_out
    mask    = offsets < total

    x_out = offsets % W_out
    y_out = (offsets // W_out) % H_out
    c_idx = (offsets // (H_out * W_out)) % C
    n_idx = offsets // (C * H_out * W_out)

    y_src = (y_out.to(tl.float32) + 0.5) * scale_h - 0.5
    x_src = (x_out.to(tl.float32) + 0.5) * scale_w - 0.5

    y0_f = tl.floor(y_src);  x0_f = tl.floor(x_src)
    y0 = y0_f.to(tl.int32);  x0 = x0_f.to(tl.int32)
    y1 = y0 + 1;              x1 = x0 + 1

    y0_c = tl.minimum(tl.maximum(y0, 0), H_in - 1)
    y1_c = tl.minimum(tl.maximum(y1, 0), H_in - 1)
    x0_c = tl.minimum(tl.maximum(x0, 0), W_in - 1)
    x1_c = tl.minimum(tl.maximum(x1, 0), W_in - 1)

    wy1 = y_src - y0_f;  wx1 = x_src - x0_f
    wy0 = 1.0 - wy1;     wx0 = 1.0 - wx1

    base  = n_idx * (C * H_in * W_in) + c_idx * (H_in * W_in)
    idx00 = base + y0_c * W_in + x0_c
    idx01 = base + y0_c * W_in + x1_c
    idx10 = base + y1_c * W_in + x0_c
    idx11 = base + y1_c * W_in + x1_c

    v00 = tl.load(x_ptr + idx00, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(x_ptr + idx01, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(x_ptr + idx10, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(x_ptr + idx11, mask=mask, other=0.0).to(tl.float32)

    result = v00 * wy0 * wx0 + v01 * wy0 * wx1 + v10 * wy1 * wx0 + v11 * wy1 * wx1
    tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)


# ── Python wrapper ──

@torch.fx.wrap
def triton_conv_bilinear_dce_f16(in_10, in_8, in_7):
    """
    Pure-Triton replacement for: conv2d(in_10, in_8, in_7) + bilinear_interpolate(512,512).
    The entire dead-code branch from the original graph is eliminated.
    """
    device = in_10.device
    # Ensure weights are on GPU and float16
    weight = in_8.to(device=device, dtype=torch.float16)  # [Cout=150, Cin=512, 1, 1]
    bias   = in_7.to(device=device, dtype=torch.float16)  # [Cout=150]

    # Input: [1, 512, 128, 128]  →  view as [Cin=512, HW=16384]
    N, Cin, H, W = in_10.shape
    Cout = weight.shape[0]          # 150
    HW   = H * W                    # 16384

    # A (weight) viewed as [M=Cout, K=Cin]
    A = weight.view(Cout, Cin)      # strides [Cin, 1]
    # B (input) viewed as [K=Cin, N=HW]   (valid because tensor is contiguous)
    B = in_10.view(Cin, HW)         # strides [HW, 1]
    # Output of GEMM: [M=Cout, N=HW]
    C = torch.empty(Cout, HW, dtype=torch.float16, device=device)

    def gemm_grid(meta):
        return (triton.cdiv(Cout, meta['BLOCK_M']) * triton.cdiv(HW, meta['BLOCK_N']),)

    gemm_bias_f16_dce_kernel[gemm_grid](
        A, B, bias, C,
        Cout, HW, Cin,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    # Reshape conv output [Cout, HW] → [1, Cout, H, W]
    conv_out = C.view(N, Cout, H, W)

    # Bilinear upsample [1, Cout, H, W] → [1, Cout, 512, 512]
    H_out, W_out = 512, 512
    total  = N * Cout * H_out * W_out
    out    = torch.empty(N, Cout, H_out, W_out, dtype=torch.float16, device=device)
    scale_h = float(H) / float(H_out)
    scale_w = float(W) / float(W_out)

    def up_grid(meta):
        return ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    bilinear_upsample_f16_dce_kernel[up_grid](
        conv_out, out,
        N, Cout, H, W, H_out, W_out,
        scale_h, scale_w,
    )
    return out


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
    """Full float16 graph including dead-code branch. Only tmp_11 is returned."""
    conv2d = torch.conv2d(in_10, in_8, in_7, (1, 1), (0, 0), (1, 1), 1)
    tmp_11 = torch.nn.functional.interpolate(conv2d, size=(512, 512), mode='bilinear', align_corners=False)
    conv2d_1 = torch.conv2d(in_9, in_6, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_13 = torch.nn.functional.batch_norm(conv2d_1, in_2, in_3, in_5, in_4, False, 0.1, 1e-05)
    tmp_14 = torch.nn.functional.relu(tmp_13, inplace=False)
    to = tmp_14.to(torch.float16)
    conv2d_2 = torch.conv2d(to, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_16 = torch.nn.functional.interpolate(conv2d_2, size=(512, 512), mode='bilinear', align_corners=False)
    return tmp_11


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
    return (in_10, in_8, in_7)


def replacement_func():
    return triton_conv_bilinear_dce_f16