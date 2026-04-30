import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
    conv2d = torch.conv2d(in_10, in_8, in_7, (1, 1), (0, 0), (1, 1), 1)
    tmp_11 = torch.nn.functional.interpolate(conv2d, size=(512, 512), mode='bilinear', align_corners=False)
    conv2d_1 = torch.conv2d(in_9, in_6, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_13 = torch.nn.functional.batch_norm(conv2d_1, in_2, in_3, in_5, in_4, False, 0.1, 1e-05)
    tmp_14 = torch.nn.functional.relu(tmp_13, inplace=False)
    to = tmp_14.to(torch.bfloat16)
    conv2d_2 = torch.conv2d(to, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_16 = torch.nn.functional.interpolate(conv2d_2, size=(512, 512), mode='bilinear', align_corners=False)
    return tmp_11


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
    return (in_10, in_8, in_7)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel_bf16(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(bias_ptr + rm, mask=rm < M, other=0.0)
    acc += bias[:, None]

    # Store
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=c_mask)


@triton.jit
def bilinear_upsample_kernel_bf16(
    input_ptr, output_ptr,
    C, H, W, OH, OW, total,
    scale_h, scale_w,
    h_lim, w_lim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    ow = offsets % OW
    temp = offsets // OW
    oh = temp % OH
    c = temp // OH

    # Source coordinates
    src_h = (oh.to(tl.float32) + 0.5) * scale_h - 0.5
    src_w = (ow.to(tl.float32) + 0.5) * scale_w - 0.5

    # Clamp
    src_h = tl.maximum(src_h, 0.0)
    src_h = tl.minimum(src_h, h_lim)
    src_w = tl.maximum(src_w, 0.0)
    src_w = tl.minimum(src_w, w_lim)

    # Floor
    h0 = src_h.to(tl.int32)
    w0 = src_w.to(tl.int32)
    h1 = h0 + 1
    w1 = w0 + 1
    # Clamp h1, w1
    h1 = tl.where(h1 > H - 1, H - 1, h1)
    w1 = tl.where(w1 > W - 1, W - 1, w1)

    # Fractional parts
    fh = src_h - h0.to(tl.float32)
    fw = src_w - w0.to(tl.float32)

    # Load 4 neighbors
    base = c * (H * W)
    v00 = tl.load(input_ptr + base + h0 * W + w0, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(input_ptr + base + h0 * W + w1, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(input_ptr + base + h1 * W + w0, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(input_ptr + base + h1 * W + w1, mask=mask, other=0.0).to(tl.float32)

    # Bilinear interpolation
    result = (1.0 - fh) * (1.0 - fw) * v00 + (1.0 - fh) * fw * v01 + fh * (1.0 - fw) * v10 + fh * fw * v11

    # Store
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def conv1x1_bilinear_upsample_bf16(input_tensor, weight, bias):
    # input_tensor: [1, C_in, H, W], weight: [C_out, C_in, 1, 1], bias: [C_out]
    C_out = weight.shape[0]
    C_in = weight.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    N_spatial = H * W

    # Allocate intermediate for conv output: [C_out, N_spatial]
    conv_output = torch.empty((C_out, N_spatial), device=input_tensor.device, dtype=input_tensor.dtype)

    # Matmul: weight[C_out, C_in] @ input[C_in, H*W] + bias[C_out]
    M, K, N = C_out, C_in, N_spatial
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    matmul_bias_kernel_bf16[grid](
        weight, input_tensor, bias, conv_output,
        M, N, K,
        C_in, 1,        # stride_am, stride_ak
        N_spatial, 1,    # stride_bk, stride_bn
        N_spatial, 1,    # stride_cm, stride_cn
    )

    # Bilinear upsampling
    OH, OW = 512, 512
    output = torch.empty((1, C_out, OH, OW), device=input_tensor.device, dtype=input_tensor.dtype)

    total_elements = C_out * OH * OW
    BLOCK_SIZE = 1024
    grid_interp = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    scale_h = float(H) / float(OH)
    scale_w = float(W) / float(OW)
    h_lim = float(H - 1)
    w_lim = float(W - 1)

    bilinear_upsample_kernel_bf16[grid_interp](
        conv_output, output,
        C_out, H, W, OH, OW, total_elements,
        scale_h, scale_w,
        h_lim, w_lim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def replacement_func():
    return conv1x1_bilinear_upsample_bf16