import os
import sys
import torch
import triton
import triton.language as tl

_PASS_DIR = os.path.dirname(__file__)
if _PASS_DIR not in sys.path:
    sys.path.append(_PASS_DIR)

from _shared_singleton_attention_flatten import singleton_attention_flatten


def pattern(bmm_1):
    tmp_4 = bmm_1.view(1, 8, 1, 32)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 256)
    return tmp_6


def replacement_args(bmm_1):
    return (bmm_1,)


def replacement_func():
    return singleton_attention_flatten