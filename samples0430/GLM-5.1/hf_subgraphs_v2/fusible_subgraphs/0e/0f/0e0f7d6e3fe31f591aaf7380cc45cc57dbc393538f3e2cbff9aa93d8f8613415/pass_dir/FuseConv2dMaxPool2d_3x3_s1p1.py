import torch
import triton
import triton.language as tl

# ============================================================
# Pattern: Conv2D(3x3, stride=1, padding=1) + MaxPool2D(3, stride=2, padding=1)
# ============================================================

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp_2,)

def replacement_args(in_0, in_1):
    return (in_0, in_1, "3x3_s1p1")


# ============================================================
# Triton Kernel for 3x3 Conv + MaxPool Fusion
# ============================================================

@triton.jit
def fused_conv_maxpool_3x3_s1p1_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, IC, H_in, W_in, OC, H_conv, W_conv, H_pool, W_pool,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr, BLOCK_IC: tl.constexpr,
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

            # Loop over IC blocks
            for ic_start in range(0, IC, BLOCK_IC):
                ic_off = ic_start + tl.arange(0, BLOCK_IC)
                ic_valid = ic_off < IC

                # For each kernel position (kh, kw) in {0,1,2}x{0,1,2}
                for kh in range(3):
                    for kw in range(3):
                        # Input position for 3x3 conv with stride=1, padding=1
                        # ih = oh * conv_stride - conv_pad + kh = oh + kh - 1
                        # iw = ow * conv_stride - conv_pad + kw = ow + kw - 1
                        ih_idx = oh_idx + kh - 1
                        iw_idx = ow_idx + kw - 1

                        # Input validity (handles zero padding)
                        ih_valid = (ih_idx >= 0) & (ih_idx < H_in)
                        iw_valid = (iw_idx >= 0) & (iw_idx < W_in)

                        # Load input: shape [BLOCK_IC, BLOCK_SP]
                        # input[n, ic, ih, iw]
                        input_offsets = (
                            n_idx[None, :] * (IC * H_in * W_in) +
                            ic_off[:, None] * (H_in * W_in) +
                            ih_idx[None, :] * W_in +
                            iw_idx[None, :]
                        )
                        input_mask = (
                            ic_valid[:, None] &
                            ih_valid[None, :] &
                            iw_valid[None, :] &
                            sp_valid[None, :] &
                            conv_valid[None, :]
                        )
                        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)

                        # Load weight: shape [BLOCK_OC, BLOCK_IC]
                        # weight[oc, ic, kh, kw]
                        weight_offsets = (
                            oc_off[:, None] * (IC * 9) +
                            ic_off[None, :] * 9 +
                            kh * 3 + kw
                        )
                        weight_mask = oc_valid[:, None] & ic_valid[None, :]
                        weight_vals = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)

                        # tl.dot: [BLOCK_OC, BLOCK_IC] @ [BLOCK_IC, BLOCK_SP] -> [BLOCK_OC, BLOCK_SP]
                        conv_vals += tl.dot(weight_vals, input_vals)

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
# Private wrapper for 3x3 variant
# ============================================================

def _fused_conv_maxpool_3x3_s1p1(weight, input):
    N, IC, H_in, W_in = input.shape
    OC = weight.shape[0]

    # Compute conv output shape: stride=1, padding=1, kernel=3x3
    H_conv = H_in  # (H_in + 2*1 - 3) // 1 + 1 = H_in
    W_conv = W_in

    # Compute pool output shape: kernel=3, stride=2, padding=1, ceil_mode=False
    H_pool = (H_conv + 2 * 1 - 3) // 2 + 1  # (H_conv - 1) // 2 + 1
    W_pool = (W_conv + 2 * 1 - 3) // 2 + 1

    output = torch.empty((N, OC, H_pool, W_pool), dtype=input.dtype, device=input.device)

    BLOCK_OC = 16
    BLOCK_SP = 64
    BLOCK_IC = 32

    num_oc_blocks = triton.cdiv(OC, BLOCK_OC)
    num_sp_blocks = triton.cdiv(N * H_pool * W_pool, BLOCK_SP)
    grid = (num_oc_blocks * num_sp_blocks,)

    fused_conv_maxpool_3x3_s1p1_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        N=N, IC=IC, H_in=H_in, W_in=W_in, OC=OC,
        H_conv=H_conv, W_conv=W_conv, H_pool=H_pool, W_pool=W_pool,
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP, BLOCK_IC=BLOCK_IC,
    )

    return output


# ============================================================
# Placeholder for 7x7 variant (never called in this pass's context)
# ============================================================

def _fused_conv_maxpool_7x7_s2p3(weight, input):
    raise NotImplementedError("7x7 variant placeholder - never called")


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