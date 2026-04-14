import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_CI': 32}),
        triton.Config({'BLOCK_CI': 64}),
        triton.Config({'BLOCK_CI': 128}),
        triton.Config({'BLOCK_CI': 256}),
        triton.Config({'BLOCK_CI': 512}),
    ],
    key=['N', 'C_in', 'C_out'],
)
@triton.jit
def fused_conv1x1_ln_relu_kernel(
    x_ptr,      # [N, C_in]
    w_ptr,      # [C_out, C_in]
    b_ptr,      # [C_out]
    ln_w_ptr,   # [C_out]
    ln_b_ptr,   # [C_out]
    out_ptr,    # [N, C_out]
    N, C_in, C_out,
    eps,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    """
    Fused kernel: 1x1 conv (linear) + layer_norm + relu
    One program per batch element n.
    Each program computes all C_out output channels, then normalizes, then relu.
    """
    n = tl.program_id(0)

    co = tl.arange(0, BLOCK_CO)
    co_mask = co < C_out

    # Accumulate linear transform result in float32
    acc = tl.zeros([BLOCK_CO], dtype=tl.float32)

    # Tiled inner loop over C_in
    for ci_start in range(0, C_in, BLOCK_CI):
        ci = ci_start + tl.arange(0, BLOCK_CI)
        ci_mask = ci < C_in

        # Load x[n, ci_start:ci_start+BLOCK_CI] -> [BLOCK_CI]
        x_vals = tl.load(x_ptr + n * C_in + ci, mask=ci_mask, other=0.0).to(tl.float32)

        # Load w[co, ci_start:ci_start+BLOCK_CI] -> [BLOCK_CO, BLOCK_CI]
        w_ptrs = w_ptr + co[:, None] * C_in + ci[None, :]
        w_vals = tl.load(w_ptrs, mask=co_mask[:, None] & ci_mask[None, :], other=0.0).to(tl.float32)

        # acc[co] += sum_k(w[co, k] * x[n, k])
        acc = acc + tl.sum(w_vals * x_vals[None, :], axis=1)

    # Add conv bias
    b_vals = tl.load(b_ptr + co, mask=co_mask, other=0.0).to(tl.float32)
    acc = acc + b_vals

    # --- Layer Norm over C_out channels ---
    # mean = sum(acc[valid]) / C_out
    acc_for_sum = tl.where(co_mask, acc, 0.0)
    mean = tl.sum(acc_for_sum) / C_out

    # variance = sum((acc - mean)^2[valid]) / C_out
    diff = tl.where(co_mask, acc - mean, 0.0)
    var = tl.sum(diff * diff) / C_out
    inv_std = 1.0 / tl.sqrt(var + eps)

    # normalize
    normalized = (acc - mean) * inv_std

    # scale and shift with layer norm params
    ln_w_vals = tl.load(ln_w_ptr + co, mask=co_mask, other=0.0).to(tl.float32)
    ln_b_vals = tl.load(ln_b_ptr + co, mask=co_mask, other=0.0).to(tl.float32)
    output = tl.where(co_mask, normalized * ln_w_vals + ln_b_vals, 0.0)

    # --- ReLU ---
    output = tl.maximum(output, 0.0)

    # Store result (cast back to original dtype)
    tl.store(out_ptr + n * C_out + co, output.to(out_ptr.dtype.element_ty), mask=co_mask)


@torch.fx.wrap
def triton_fused_conv1x1_ln_relu(in_0, in_1, in_2, in_3, in_4):
    """
    in_0: conv bias     [C_out]
    in_1: conv weight   [C_out, C_in, 1, 1]
    in_2: ln bias       [C_out, 1, 1]
    in_3: ln weight     [C_out, 1, 1]
    in_4: input         [N, C_in, 1, 1]
    """
    N = in_4.shape[0]
    C_out = in_1.shape[0]
    C_in = in_1.shape[1]

    out = torch.empty((N, C_out, 1, 1), dtype=in_4.dtype, device=in_4.device)

    BLOCK_CO = triton.next_power_of_2(C_out)

    # Flatten spatial dims (all 1x1) for kernel
    x_flat = in_4.contiguous().view(N, C_in)
    w_flat = in_1.contiguous().view(C_out, C_in)
    ln_w_flat = in_3.contiguous().view(C_out)
    ln_b_flat = in_2.contiguous().view(C_out)
    out_flat = out.view(N, C_out)

    fused_conv1x1_ln_relu_kernel[(N,)](
        x_flat, w_flat, in_0, ln_w_flat, ln_b_flat, out_flat,
        N, C_in, C_out,
        1e-5,
        BLOCK_CO=BLOCK_CO,
    )

    return out