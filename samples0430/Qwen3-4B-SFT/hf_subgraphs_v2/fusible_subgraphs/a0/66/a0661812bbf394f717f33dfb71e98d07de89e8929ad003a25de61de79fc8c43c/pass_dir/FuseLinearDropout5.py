"""
Pass: FuseLinearDropout5
Matches: F.linear + F.dropout(p=0.05, training=False)  →  returns linear result only (single output)
Covers: bfloat16/Aniemore_unispeech-sat-resd (dropout=0.05)
        and float16/tiny-random-UniSpeechSatForSequenceClassification (dropout=0.05)
        both with output order (transposed-first, linear-second).
"""
import torch
import triton
import triton.language as tl
from pass_dir.FuseLinearDropout import _triton_linear, _linear_bias_kernel


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    result = torch.nn.functional.dropout(linear, 0.05, False, False)
    return result


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return _triton_linear