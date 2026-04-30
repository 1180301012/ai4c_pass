import os
import sys
import torch
import triton
import triton.language as tl

_PASS_DIR = os.path.dirname(__file__)
if _PASS_DIR not in sys.path:
    sys.path.append(_PASS_DIR)

from _shared_singleton_attention_flatten import singleton_attention_flatten


def pattern(scores, in_2):
    tmp_1 = torch.nn.functional.softmax(scores, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    bmm_1 = torch.bmm(tmp_2, in_2)
    tmp_4 = bmm_1.view(1, 8, 1, 32)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 256)
    return tmp_6


def replacement_args(scores, in_2):
    return (in_2,)


def replacement_func():
    return singleton_attention_flatten