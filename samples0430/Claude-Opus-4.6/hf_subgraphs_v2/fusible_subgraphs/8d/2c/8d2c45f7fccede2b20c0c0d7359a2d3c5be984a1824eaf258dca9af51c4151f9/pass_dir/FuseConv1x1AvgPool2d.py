import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match conv2d(1x1) + avg_pool2d(2x2, stride=2) pattern.
    in_0: weight [C_out, C_in, 1, 1]
    in_1: input [N, C_in, H, W]
    """
    conv = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    pool = torch.nn.functional.avg_pool2d(conv, 2, 2, 0, False, True, None)
    return pool


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_avgpool2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    out_HW,
    out_W,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_w_o, stride_w_i,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused 1x1 convolution + 2x2 average pooling kernel.
    
    Exploits: avg_pool2d(conv1x1(x)) = conv1x1(avg_pool2d(x))
    Fuses both ops, reducing compute by 4x and eliminating intermediate buffer.
    
    M = batch * out_H * out_W (output spatial positions)
    N = C_out (output channels)
    K = C_in (input channels / reduction dimension)
    """
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decompose m into (batch_idx, oh, ow)
    batch_idx = offs_m // out_HW
    spatial = offs_m % out_HW
    oh = spatial // out_W
    ow = spatial % out_W

    # Compute input spatial positions for 2x2 pooling window
    ih0 = oh * 2
    iw0 = ow * 2

    # Base input offsets for 4 pool positions [BLOCK_M]
    base_00 = batch_idx * stride_in_n + ih0 * stride_in_h + iw0 * stride_in_w
    base_01 = base_00 + stride_in_w
    base_10 = base_00 + stride_in_h
    base_11 = base_00 + stride_in_h + stride_in_w

    # Initialize float32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # Main reduction loop over input channels (K = C_in)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        mk_mask = m_mask[:, None] & k_mask[None, :]

        # Channel offset for input loading [1, BLOCK_K]
        chan_offset = k_offs[None, :] * stride_in_c

        # Load 4 input values for 2x2 pooling [BLOCK_M, BLOCK_K]
        a00 = tl.load(input_ptr + base_00[:, None] + chan_offset, mask=mk_mask, other=0.0)
        a01 = tl.load(input_ptr + base_01[:, None] + chan_offset, mask=mk_mask, other=0.0)
        a10 = tl.load(input_ptr + base_10[:, None] + chan_offset, mask=mk_mask, other=0.0)
        a11 = tl.load(input_ptr + base_11[:, None] + chan_offset, mask=mk_mask, other=0.0)

        # Compute 2x2 average
        a = (a00 + a01 + a10 + a11) * 0.25

        # Load weight tile [BLOCK_K, BLOCK_N]
        # weight[co, ci] -> need weight.T[ci, co] for matmul
        nk_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(weight_ptr + k_offs[:, None] * stride_w_i + offs_n[None, :] * stride_w_o,
                    mask=nk_mask, other=0.0)

        # Matrix multiply: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, b)

    # Store output with proper dtype conversion
    out_offset = (batch_idx[:, None] * stride_out_n + offs_n[None, :] * stride_out_c +
                  oh[:, None] * stride_out_h + ow[:, None] * stride_out_w)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(output_ptr + out_offset, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def fused_conv1x1_avgpool2d(weight, input_tensor):
    """
    Fused replacement for conv2d(1x1) + avg_pool2d(2x2, stride=2).
    """
    batch = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    C_out = weight.shape[0]
    out_H = H // 2
    out_W = W // 2

    M = batch * out_H * out_W
    N = C_out
    K = C_in

    output = torch.empty((batch, C_out, out_H, out_W), dtype=input_tensor.dtype, device=input_tensor.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    fused_conv1x1_avgpool2d_kernel[grid](
        input_tensor, weight, output,
        M, N, K,
        out_H * out_W,
        out_W,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )

    return output


def replacement_func():
    return fused_conv1x1_avgpool2d