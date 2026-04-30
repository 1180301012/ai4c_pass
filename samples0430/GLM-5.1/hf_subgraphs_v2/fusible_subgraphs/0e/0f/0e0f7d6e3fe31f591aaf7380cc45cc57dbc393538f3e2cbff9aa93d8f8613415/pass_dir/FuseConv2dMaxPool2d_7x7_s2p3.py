import torch
import triton
import triton.language as tl

# ============================================================
# Pattern: Conv2D(7x7, stride=2, padding=3) + MaxPool2D(3, stride=2, padding=1)
# ============================================================

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_3 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp_3,)

def replacement_args(in_0, in_1):
    return (in_0, in_1, "7x7_s2p3")


# ============================================================
# Triton Kernel for 7x7 Conv + MaxPool Fusion
# IC=3, KH=KW=7, stride=2, padding=3
# Uses element-wise computation (IC too small for tl.dot)
# ============================================================

@triton.jit
def fused_conv_maxpool_7x7_s2p3_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, IC, H_in, W_in, OC, H_conv, W_conv, H_pool, W_pool,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid = tl.program_id(0)

    num_oc_blocks = tl.cdiv(OC, BLOCK_OC)
    num_sp_blocks = tl.cdiv(N * H_pool * W_pool, BLOCK_SP)

    pid_oc = pid // num_sp_blocks
    pid_sp = pid % num_sp_blocks

    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_off = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    # Decode spatial position: sp -> (n, ph, pw)
    pool_hw = H_pool * W_pool
    n_idx = sp_off // pool_hw
    ph_idx = (sp_off % pool_hw) // W_pool
    pw_idx = (sp_off % pool_hw) % W_pool

    # Spatial validity mask
    sp_valid = (n_idx < N) & (ph_idx < H_pool) & (pw_idx < W_pool)

    # OC validity mask
    oc_valid = oc_off < OC

    # Initialize max values to -inf
    max_vals = tl.full([BLOCK_OC, BLOCK_SP], float('-inf'), dtype=tl.float32)

    # For each pool window position (pkh, pkw) in {0,1,2}x{0,1,2}
    for pkh in range(3):
        for pkw in range(3):
            # Conv output position for this pool window position
            # oh = ph * pool_stride - pool_pad + pkh = ph*2 - 1 + pkh
            # ow = pw * pool_stride - pool_pad + pkw = pw*2 - 1 + pkw
            oh_idx = ph_idx * 2 - 1 + pkh
            ow_idx = pw_idx * 2 - 1 + pkw

            # Conv position validity (for max_pool padding handling)
            conv_valid = (oh_idx >= 0) & (oh_idx < H_conv) & (ow_idx >= 0) & (ow_idx < W_conv) & sp_valid

            # Conv accumulator (float32 for precision)
            conv_vals = tl.zeros([BLOCK_OC, BLOCK_SP], dtype=tl.float32)

            # For each input channel (IC=3)
            for ic in range(3):
                # For each kernel position (kh, kw) in 7x7
                for kh in range(7):
                    for kw in range(7):
                        # Input position for 7x7 conv with stride=2, padding=3
                        # ih = oh * conv_stride - conv_pad + kh = oh*2 - 3 + kh
                        # iw = ow * conv_stride - conv_pad + kw = ow*2 - 3 + kw
                        ih_idx = oh_idx * 2 - 3 + kh
                        iw_idx = ow_idx * 2 - 3 + kw

                        # Input validity (handles zero padding)
                        ih_valid = (ih_idx >= 0) & (ih_idx < H_in)
                        iw_valid = (iw_idx >= 0) & (iw_idx < W_in)

                        # Load input: shape [BLOCK_SP]
                        # input[n, ic, ih, iw]
                        input_offset = (
                            n_idx * (IC * H_in * W_in) +
                            ic * (H_in * W_in) +
                            ih_idx * W_in +
                            iw_idx
                        )
                        input_mask = ih_valid & iw_valid & sp_valid & conv_valid
                        input_val = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)

                        # Load weight: shape [BLOCK_OC]
                        # weight[oc, ic, kh, kw]
                        weight_offset = oc_off * (IC * 49) + ic * 49 + kh * 7 + kw
                        weight_mask = oc_valid
                        weight_val = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)

                        # Accumulate: [BLOCK_OC, BLOCK_SP] += weight[oc] * input[sp]
                        conv_vals += weight_val[:, None] * input_val[None, :].to(tl.float32)

            # Override invalid conv positions with -inf (for max_pool padding)
            conv_vals_final = tl.where(conv_valid[None, :] & oc_valid[:, None], conv_vals, float('-inf'))

            # Update max
            max_vals = tl.maximum(max_vals, conv_vals_final)

    # Store output: shape [BLOCK_OC, BLOCK_SP]
    # output[n, oc, ph, pw]
    output_offsets = (
        n_idx[None, :] * (OC * H_pool * W_pool) +
        oc_off[:, None] * (H_pool * W_pool) +
        ph_idx[None, :] * W_pool +
        pw_idx[None, :]
    )
    output_mask = oc_valid[:, None] & sp_valid[None, :]
    tl.store(output_ptr + output_offsets, max_vals, mask=output_mask)


# ============================================================
# Private wrapper for 7x7 variant
# ============================================================

def _fused_conv_maxpool_7x7_s2p3(weight, input):
    N, IC, H_in, W_in = input.shape
    OC = weight.shape[0]

    # Compute conv output shape: stride=2, padding=3, kernel=7x7
    H_conv = (H_in + 2 * 3 - 7) // 2 + 1  # (H_in - 1) // 2 + 1
    W_conv = (W_in + 2 * 3 - 7) // 2 + 1

    # Compute pool output shape: kernel=3, stride=2, padding=1, ceil_mode=False
    H_pool = (H_conv + 2 * 1 - 3) // 2 + 1  # (H_conv - 1) // 2 + 1
    W_pool = (W_conv + 2 * 1 - 3) // 2 + 1

    output = torch.empty((N, OC, H_pool, W_pool), dtype=input.dtype, device=input.device)

    BLOCK_OC = 16
    BLOCK_SP = 64

    num_oc_blocks = triton.cdiv(OC, BLOCK_OC)
    num_sp_blocks = triton.cdiv(N * H_pool * W_pool, BLOCK_SP)
    grid = (num_oc_blocks * num_sp_blocks,)

    fused_conv_maxpool_7x7_s2p3_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        N=N, IC=IC, H_in=H_in, W_in=W_in, OC=OC,
        H_conv=H_conv, W_conv=W_conv, H_pool=H_pool, W_pool=W_pool,
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
    )

    return output


# ============================================================
# Placeholder for 3x3 variant (never called in this pass's context)
# ============================================================

def _fused_conv_maxpool_3x3_s1p1(weight, input):
    raise NotImplementedError("3x3 variant placeholder - never called")


# ============================================================
# Shared dispatch wrapper (MUST be identical across all pass files)
# ============================================================

@torch.fx.wrap
def fused_conv_maxpool_dispatch(weight, input, route):
    if route == "3x3_s1p1":
        return _fused_conv_maxpool_3x3_s1p1(weight, input)
    elif route == "7x7_s2p3":
        return _fused_conv_maxpool_7x7_s2p3(weight, input)
    else:
        raise ValueError(f"Unknown route: {route}")


# ============================================================
# Replacement function (returns the shared dispatch wrapper)
# ============================================================

def replacement_func():
    return fused_conv_maxpool_dispatch