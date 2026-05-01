import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_CIN': 32}, num_warps=2),
        triton.Config({'BLOCK_CIN': 64}, num_warps=4),
        triton.Config({'BLOCK_CIN': 128}, num_warps=4),
        triton.Config({'BLOCK_CIN': 256}, num_warps=4),
        triton.Config({'BLOCK_CIN': 512}, num_warps=8),
        triton.Config({'BLOCK_CIN': 1024}, num_warps=8),
    ],
    key=['C_in', 'C_out', 'BLOCK_COUT']
)
@triton.jit
def _fused_conv_ln_relu_kernel(
    inp_ptr, w_ptr, cbias_ptr, ln_w_ptr, ln_b_ptr, out_ptr,
    N, C_in, C_out, eps,
    BLOCK_CIN: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    """
    Fused 1x1 conv (GEMM) + LayerNorm + ReLU kernel.
    Each program handles one batch element.

    inp:    [N, C_in]  (1x1 spatial squeezed)
    w:      [C_out, C_in]
    cbias:  [C_out]
    ln_w:   [C_out]  (layer_norm weight, originally [C_out,1,1])
    ln_b:   [C_out]  (layer_norm bias,   originally [C_out,1,1])
    out:    [N, C_out]
    """
    batch = tl.program_id(0)

    cout_idx = tl.arange(0, BLOCK_COUT)   # [BLOCK_COUT]
    mask_cout = cout_idx < C_out

    # Accumulator in float32 for numerical stability
    acc = tl.zeros([BLOCK_COUT], dtype=tl.float32)

    # Tile over C_in to compute GEMM
    for cin_start in range(0, C_in, BLOCK_CIN):
        cin_idx = cin_start + tl.arange(0, BLOCK_CIN)
        mask_cin = cin_idx < C_in

        # Load input vector for this batch element: shape [BLOCK_CIN]
        x = tl.load(inp_ptr + batch * C_in + cin_idx,
                    mask=mask_cin, other=0.0).to(tl.float32)

        # Load weight tile: shape [BLOCK_COUT, BLOCK_CIN]
        w_ptrs = w_ptr + cout_idx[:, None] * C_in + cin_idx[None, :]
        w_mask = mask_cout[:, None] & mask_cin[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # acc[i] += sum_j w[i,j] * x[j]
        acc += tl.sum(w * x[None, :], axis=1)

    # Add conv bias
    cbias = tl.load(cbias_ptr + cout_idx, mask=mask_cout, other=0.0).to(tl.float32)
    acc = tl.where(mask_cout, acc + cbias, 0.0)

    # ------- LayerNorm -------
    # Mean over C_out valid elements
    sum_acc = tl.sum(tl.where(mask_cout, acc, 0.0), axis=0)
    mean = sum_acc / C_out

    # Variance
    diff = tl.where(mask_cout, acc - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C_out
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Affine transform
    ln_w = tl.load(ln_w_ptr + cout_idx, mask=mask_cout, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + cout_idx, mask=mask_cout, other=0.0).to(tl.float32)

    normalized = diff * inv_std * ln_w + ln_b

    # ------- ReLU -------
    output = tl.maximum(normalized, 0.0)

    # Store float32 result (wrapper will cast to original dtype if needed)
    tl.store(out_ptr + batch * C_out + cout_idx, output, mask=mask_cout)


@torch.fx.wrap
def fused_conv_ln_relu(in_0, in_1, in_2, in_3, in_4):
    """
    Fused conv2d + layer_norm + relu for 1x1 spatial tensors.

    in_0 : conv_bias   [C_out]
    in_1 : conv_weight [C_out, C_in, 1, 1]
    in_2 : ln_bias     [C_out, 1, 1]
    in_3 : ln_weight   [C_out, 1, 1]
    in_4 : input       [N, C_in, 1, 1]

    Returns:
        output [N, C_out, 1, 1] with same dtype as in_4
    """
    N    = in_4.shape[0]
    C_in  = in_4.shape[1]
    C_out = in_1.shape[0]

    # Smallest power-of-2 >= C_out (needed as constexpr tl.arange bound)
    BLOCK_COUT = 1
    while BLOCK_COUT < C_out:
        BLOCK_COUT *= 2

    # Always accumulate in float32 for numerical stability
    output_f32 = torch.empty((N, C_out, 1, 1), dtype=torch.float32, device=in_4.device)

    _fused_conv_ln_relu_kernel[(N,)](
        in_4,        # inp_ptr   [N, C_in, 1, 1]  — contiguous, stride C_in per batch
        in_1,        # w_ptr     [C_out, C_in, 1, 1]
        in_0,        # cbias_ptr [C_out]
        in_3,        # ln_w_ptr  [C_out, 1, 1] — contiguous as C_out elements
        in_2,        # ln_b_ptr  [C_out, 1, 1]
        output_f32,  # out_ptr   [N, C_out, 1, 1]
        N, C_in, C_out,
        1e-5,
        BLOCK_COUT=BLOCK_COUT,
    )

    # Cast back to original dtype if needed
    if in_4.dtype != torch.float32:
        return output_f32.to(in_4.dtype)
    return output_f32