import torch
from pass_dir.se_kernels import triton_fused_se_dispatch


# ─── Pattern ────────────────────────────────────────────────────────────────
# Matches:  conv2d → +3.0 → /6.0 → clamp_(0,1) → in_2 * attn
# Hard-Sigmoid (ReLU6 variant) used in MobileNetV3 float32.
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 3.0
    tmp_4 = tmp_3 / 6.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    # Append route tag so the shared dispatch wrapper knows which constants to use.
    return (in_0, in_1, in_2, in_3, "36")


def replacement_func():
    return triton_fused_se_dispatch