import torch
import triton
import triton.language as tl


@triton.jit
def full_fused_kernel(
    in2_ptr,     # in_2 [1, 2, 1, 8] → 16 values
    weight_ptr,  # in_1 [128, 2, 1, 8] → 128*16 values
    bias_ptr,    # in_0 [128] → 128 values
    in3_ptr,     # in_3 [1, 2, 8, 8] → 128 values
    out4_ptr,    # tmp_4 [1, 2, 8, 8] → 128 values
    out6_ptr,    # tmp_6 [1, 2, 8, 8] → 128 values
    C_IN: tl.constexpr,       # 16
    OC:   tl.constexpr,       # 128
    BLOCK_HW: tl.constexpr,   # 8
):
    """
    Grid = (16,).  One program per (batch=0, c, h) row.
    Each program computes:
      1. conv+sigmoid for all 128 output channels (loop over OC)
         → 128 * sigmoid(bias + dot(in2[0:16], weight[oc, :])) stored to out4
      2. row-normalisation for in_3 (single row)
         → in3[row*8:row*8+8] / sum(in3[row*8:row*8+8]) stored to out6
    """
    row_idx = tl.program_id(0)   # [0, 16)
    hw_offsets = tl.arange(0, BLOCK_HW)   # [0, 1, ..., 7]

    # ── Part 1: conv + sigmoid, loop over all OC=128 channels ────────── #
    cin_offsets = tl.arange(0, C_IN)   # [0, 1, ..., 15]

    for oc in range(OC):               # compile-time unroll: 128 iterations
        # scalar load: in2[flat_idx] for each flat_idx in cin_offsets
        in2_vals = tl.load(in2_ptr + cin_offsets).to(tl.float32)  # [16]
        w_offsets = oc * C_IN + cin_offsets                        # [16]
        w_vals = tl.load(weight_ptr + w_offsets).to(tl.float32)   # [16]
        bias_val = tl.load(bias_ptr + oc).to(tl.float32)          # scalar
        dot = tl.sum(in2_vals * w_vals, axis=0)                   # scalar
        sig_val = tl.sigmoid(dot + bias_val)                       # scalar

        # Map oc → flat index in [1,2,8,8]:  oc = c_out*8 + h_idx
        c_out  = oc // 8
        h_idx  = oc % 8
        out4_flat = c_out * 64 + h_idx * 8 + hw_offsets            # [8]
        # Broadcast scalar sig_val → [8] and store
        sig_broadcast = tl.full((BLOCK_HW,), sig_val, tl.float32)
        tl.store(out4_ptr + out4_flat, sig_broadcast)

    # ── Part 2: row normalisation from in_3 ──────────────────────────── #
    in3_base  = row_idx * BLOCK_HW
    in3_vals  = tl.load(in3_ptr + in3_base + hw_offsets).to(tl.float32)  # [8]
    row_sum   = tl.sum(in3_vals, axis=0)                                  # scalar
    out6_vals = in3_vals / row_sum                                        # [8]
    tl.store(out6_ptr + in3_base + hw_offsets, out6_vals)


def pattern(in_0, in_1, in_2, in_3):
    """
    Matches the ENTIRE forward:
        conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_3  = conv2d.view(1, 2, 8, 8)
        tmp_4  = tmp_3.sigmoid()
        tmp_5  = in_3.sum(dim=3, keepdim=True)
        tmp_6  = in_3 / tmp_5
    Returns both observable values (tmp_6, tmp_4).
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(1, 2, 8, 8)
    tmp_4 = tmp_3.sigmoid()
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return (tmp_6, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def full_fused_wrapper(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [128]
    in_1 : weight [128, 2, 1, 8]
    in_2 : input  [1,  2, 1, 8]   on CUDA
    in_3 : attn   [1,  2, 8, 8]   on CUDA
    Returns: (tmp_6 [1,2,8,8], tmp_4 [1,2,8,8])
    """
    BLOCK_HW = 8

    out4 = torch.empty((1, 2, 8, 8), dtype=in_2.dtype, device=in_2.device)
    out6 = torch.empty((1, 2, 8, 8), dtype=in_2.dtype, device=in_2.device)

    # Grid = (16,) — one program handles all 16 (batch=0, c, h) rows
    full_fused_kernel[(16,)](
        in_2, in_1, in_0, in_3,
        out4, out6,
        C_IN=16,
        OC=128,
        BLOCK_HW=BLOCK_HW,
    )

    return (out6, out4)


def replacement_func():
    return full_fused_wrapper