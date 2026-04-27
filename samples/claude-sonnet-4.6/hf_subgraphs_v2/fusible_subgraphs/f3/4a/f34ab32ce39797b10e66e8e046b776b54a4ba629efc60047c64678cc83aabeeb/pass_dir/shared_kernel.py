import torch
import triton
import triton.language as tl


@triton.jit
def _fused_conv1x1_ln_relu_kernel(
    x_ptr,      # [N, C_in, 1, 1]  input
    w_ptr,      # [C_out, C_in, 1, 1]  conv weight
    cb_ptr,     # [C_out]  conv bias
    ln_w_ptr,   # [C_out, 1, 1]  layer norm weight (gamma)
    ln_b_ptr,   # [C_out, 1, 1]  layer norm bias (beta)
    out_ptr,    # [N, C_out, 1, 1]  output
    N, C_in, C_out,
    BLOCK_COUT: tl.constexpr,
    BLOCK_CIN: tl.constexpr,
):
    """
    Fused kernel: conv1x1 (matmul) + layer_norm + relu.
    One program per sample n. Handles all C_out channels in BLOCK_COUT-width tiles.
    """
    n = tl.program_id(0)

    c_idx = tl.arange(0, BLOCK_COUT)
    c_mask = c_idx < C_out
    # Clamp to avoid OOB loads for padding channels
    c_clamped = tl.where(c_mask, c_idx, 0)

    # ---- Step 1: Compute conv output (matmul + bias) in float32 ----
    acc = tl.zeros((BLOCK_COUT,), dtype=tl.float32)

    for k_start in range(0, C_in, BLOCK_CIN):
        k_idx = k_start + tl.arange(0, BLOCK_CIN)
        k_mask = k_idx < C_in
        k_clamped = tl.where(k_mask, k_idx, 0)

        # Load x[n, k]: contiguous along k since H=W=1
        x_vals = tl.load(
            x_ptr + n * C_in + k_clamped,
            mask=k_mask, other=0.0
        ).to(tl.float32)

        # Load w[c, k]: w has strides (C_in, 1, 1, 1), so w[c,k] = w_ptr + c*C_in + k
        w_vals = tl.load(
            w_ptr + c_clamped[:, None] * C_in + k_clamped[None, :],
            mask=c_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)

        # acc[c] += sum_k w[c,k] * x[k]
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Add conv bias
    cb_vals = tl.load(cb_ptr + c_clamped, mask=c_mask, other=0.0).to(tl.float32)
    acc += cb_vals

    # ---- Step 2: Layer norm over C_out channels ----
    # mean = sum(acc[valid]) / C_out
    acc_valid = tl.where(c_mask, acc, 0.0)
    mean = tl.sum(acc_valid) / C_out

    # variance = sum((acc - mean)^2 [valid]) / C_out
    centered = acc - mean
    centered_sq = tl.where(c_mask, centered * centered, 0.0)
    var = tl.sum(centered_sq) / C_out
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    # Apply scale (ln_w) and shift (ln_b); shapes are [C_out,1,1] contiguous
    ln_w = tl.load(ln_w_ptr + c_clamped, mask=c_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + c_clamped, mask=c_mask, other=0.0).to(tl.float32)
    normed = centered * inv_std * ln_w + ln_b

    # ---- Step 3: ReLU ----
    out_val = tl.maximum(normed, 0.0)

    # ---- Step 4: Store to [N, C_out, 1, 1] ----
    # strides: (C_out, 1, 1, 1), so out[n,c,0,0] = out_ptr + n*C_out + c
    tl.store(out_ptr + n * C_out + c_idx, out_val, mask=c_mask)


@torch.fx.wrap
def fused_conv1x1_ln_relu(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : conv bias        [C_out]
    in_1 : conv weight      [C_out, C_in, 1, 1]
    in_2 : ln bias (beta)   [C_out, 1, 1]
    in_3 : ln weight (gamma)[C_out, 1, 1]
    in_4 : input            [N, C_in, 1, 1]
    Returns [N, C_out, 1, 1] after conv1x1 -> layer_norm -> relu
    """
    N    = in_4.shape[0]
    C_in = in_1.shape[1]
    C_out = in_1.shape[0]

    # Round up to next power-of-2 for the constexpr block size
    BLOCK_COUT = triton.next_power_of_2(C_out)
    BLOCK_COUT = max(BLOCK_COUT, 16)   # minimum 16

    # Choose BLOCK_CIN to tile the reduction; cap at 512
    BLOCK_CIN = triton.next_power_of_2(min(C_in, 512))
    BLOCK_CIN = max(BLOCK_CIN, 32)

    out = torch.empty((N, C_out, 1, 1), dtype=in_4.dtype, device=in_4.device)

    grid = (N,)
    _fused_conv1x1_ln_relu_kernel[grid](
        in_4, in_1, in_0, in_3, in_2, out,
        N, C_in, C_out,
        BLOCK_COUT=BLOCK_COUT,
        BLOCK_CIN=BLOCK_CIN,
    )

    return out