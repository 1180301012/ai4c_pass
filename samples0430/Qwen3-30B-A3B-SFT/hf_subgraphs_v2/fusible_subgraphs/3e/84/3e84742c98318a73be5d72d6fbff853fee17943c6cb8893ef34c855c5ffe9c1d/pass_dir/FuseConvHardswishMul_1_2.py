"""
Fused pass A: fuse hardswish(+1.0/2.0) broadcast-multiply into a single Triton kernel.
Pattern matches the post-conv elementwise chain: add(1.0) -> div(2.0) -> clamp(0,1) -> mul.
Let cuBLAS handle conv2d; only the bandwidth-bound elementwise broadcast multiply is fused.
Handles bfloat16 and float16 S-ViPNAS-MobileNetV3 graphs.
"""
import torch
import triton
import triton.language as tl

from pass_dir._shared_hardswish_kernels import _dispatch_hardswish_mul


# ── Pattern & routing helpers ────────────────────────────────────────────────

def pattern(conv_out, in_2):
    tmp_3 = conv_out + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


def replacement_args(conv_out, in_2):
    return (conv_out, in_2, "r1_2")


def replacement_func():
    return _dispatch_hardswish_mul