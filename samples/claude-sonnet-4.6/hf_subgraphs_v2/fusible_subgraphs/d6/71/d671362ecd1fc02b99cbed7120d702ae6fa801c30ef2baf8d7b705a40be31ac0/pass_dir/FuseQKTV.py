import torch
from pass_dir.shared_qkv import shared_dispatch  # same object across all passes


# ──────────────────────────────────────────────────────────────────────────────
# Pass: fuse split([32,32,128]) + 3×permute(0,2,1,3) + transpose(-2,-1)
# into a single Triton kernel that reads from the reshaped linear output
# tmp_4 (B, 49, 8, 192) and directly writes Q, K^T, V.
#
# This eliminates all three separate kernel launches and the 3 output buffers
# from FuseSplitPermuteTranspose + FusePermute0213, reducing to ONE kernel.
# ──────────────────────────────────────────────────────────────────────────────

def pattern(tmp_4):
    """
    Multi-output pattern: matches the full post-linear rearrangement.
    tmp_4 is the (B,49,8,192) contiguous reshape of the linear output.
    Returns the three observable outputs: Q, K^T, V.
    """
    split  = tmp_4.split([32, 32, 128], dim=3)
    tmp_6  = split[0]
    tmp_7  = split[1]
    tmp_8  = split[2]
    tmp_9  = tmp_6.permute(0, 2, 1, 3)
    tmp_10 = tmp_7.permute(0, 2, 1, 3)
    tmp_11 = tmp_8.permute(0, 2, 1, 3)
    tmp_13 = tmp_10.transpose(-2, -1)
    return (tmp_9, tmp_13, tmp_11)


def replacement_args(tmp_4):
    # Route "qktv" → single-kernel Q/K^T/V computation
    return (tmp_4, "qktv")


def replacement_func():
    return shared_dispatch