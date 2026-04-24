import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_C_OUT': 4, 'BLOCK_C_IN': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_C_OUT': 4, 'BLOCK_C_IN': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C_OUT': 4, 'BLOCK_C_IN': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C_OUT': 4, 'BLOCK_C_IN': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C_OUT': 4, 'BLOCK_C_IN': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_C_OUT': 1, 'BLOCK_C_IN': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_C_OUT': 1, 'BLOCK_C_IN': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C_OUT': 1, 'BLOCK_C_IN': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C_OUT': 1, 'BLOCK_C_IN': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C_OUT': 1, 'BLOCK_C_IN': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_C_OUT': 16, 'BLOCK_C_IN': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_C_OUT': 16, 'BLOCK_C_IN': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C_OUT': 16, 'BLOCK_C_IN': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C_OUT': 16, 'BLOCK_C_IN': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C_OUT': 16, 'BLOCK_C_IN': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C_OUT': 64, 'BLOCK_C_IN': 256}, num_warps=8),
    ],
    key=['M', 'C_out', 'DTYPE'],
)
@triton.jit
def _conv1x1_perm_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C_in, C_out, H, W,
    M,  # B * H * W
    BLOCK_M: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """
    Fused 1x1 conv (NCHW) + permute(0,2,3,1) + reshape(B,-1,C_out) + sigmoid.

    input_ptr  : [B, C_in, H, W] NCHW contiguous
    weight_ptr : [C_out, C_in]   (flattened 1x1 kernel)
    bias_ptr   : [C_out]
    output_ptr : [B, H*W, C_out]
    """
    pid_m = tl.program_id(0)   # tile over spatial dim M = B*H*W
    pid_co = tl.program_id(1)  # tile over output channels

    m_start = pid_m * BLOCK_M
    co_start = pid_co * BLOCK_C_OUT

    m_idx = m_start + tl.arange(0, BLOCK_M)    # [BLOCK_M]
    co_idx = co_start + tl.arange(0, BLOCK_C_OUT)  # [BLOCK_C_OUT]

    # Decode m = b*H*W + h*W + w  into (b_idx, hw_idx)
    hw_idx = m_idx % (H * W)    # position in spatial grid
    b_idx = m_idx // (H * W)    # batch index

    hw_mask = hw_idx < (H * W)
    m_mask = m_idx < M
    co_mask = co_idx < C_out

    # Accumulate in fp32 for numerical stability
    acc = tl.zeros([BLOCK_M, BLOCK_C_OUT], dtype=tl.float32)

    # H*W = 64*128 = 8192 for all graphs
    hw_pos = hw_idx[:, None]  # [BLOCK_M, 1]
    w_pos = hw_pos % W        # [BLOCK_M, 1]
    h_pos = hw_pos // W       # [BLOCK_M, 1]
    b_pos = b_idx[:, None]    # [BLOCK_M, 1]

    for k_start in range(0, C_in, BLOCK_C_IN):
        k_idx = k_start + tl.arange(0, BLOCK_C_IN)  # [BLOCK_C_IN]
        k_mask = k_idx < C_in

        # Load input block: shape [BLOCK_M, BLOCK_C_IN]
        # input[b, ic, h, w] -> offset b*(C_in*H*W) + ic*(H*W) + h*W + w
        a_offset = (b_pos * (C_in * H * W)
                    + k_idx[None, :] * (H * W)
                    + h_pos * W
                    + w_pos)  # [BLOCK_M, BLOCK_C_IN]
        a = tl.load(
            input_ptr + a_offset,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Load weight block: shape [BLOCK_C_OUT, BLOCK_C_IN]
        w_offset = co_idx[:, None] * C_in + k_idx[None, :]  # [BLOCK_C_OUT, BLOCK_C_IN]
        w = tl.load(
            weight_ptr + w_offset,
            mask=co_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # acc += a @ w.T  =>  [BLOCK_M, BLOCK_C_OUT]
        acc += tl.sum(
            a[:, None, :] * w[None, :, :],
            axis=2,
        )

    # Add bias
    bias = tl.load(bias_ptr + co_idx, mask=co_mask, other=0.0)  # [BLOCK_C_OUT]
    acc += bias[None, :]

    # Sigmoid: σ(x) = 1 / (1 + exp(-x))
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # Convert back to the original dtype
    out_val = acc.to(DTYPE)

    # Store output: [B, H*W, C_out] contiguous row-major
    out_offset = (b_idx[:, None] * (H * W * C_out)
                  + hw_idx[:, None] * C_out
                  + co_idx[None, :])  # [BLOCK_M, BLOCK_C_OUT]
    tl.store(
        output_ptr + out_offset,
        out_val,
        mask=m_mask[:, None] & co_mask[None, :],
    )


@torch.fx.wrap
def fused_conv_permute_sigmoid(in_0, in_1, in_2, route):
    """
    in_0 : bias   [C_out]
    in_1 : weight [C_out, C_in, 1, 1]
    in_2 : input  [B, C_in, H, W]
    route: dispatch string (not used for computation, only to select pattern)

    Returns: sigmoid(permute(conv1x1(in_2, in_1, in_0))) reshaped to [B, H*W, C_out]
    """
    B = in_2.shape[0]
    C_in = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    C_out = in_1.shape[0]
    M = B * H * W

    output = torch.empty((B, H * W, C_out), dtype=in_2.dtype, device=in_2.device)

    # Map torch dtype -> triton dtype string for constexpr dispatch
    dtype_str = {
        torch.float16: 'float16',
        torch.bfloat16: 'bfloat16',
        torch.float32: 'float32',
    }[in_2.dtype]

    # BLOCK_C_OUT must be a power of 2 for tl.arange; larger blocks cover small C_out with masking
    if C_out <= 1:
        BLOCK_C_OUT_val = 1
    elif C_out <= 4:
        BLOCK_C_OUT_val = 4
    elif C_out <= 16:
        BLOCK_C_OUT_val = 16
    else:
        BLOCK_C_OUT_val = 64

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_C_OUT']),
    )

    _conv1x1_perm_sigmoid_kernel[grid](
        in_2, in_1, in_0, output,
        B, C_in, C_out, H, W,
        M,
        DTYPE=dtype_str,
    )

    return (output,)