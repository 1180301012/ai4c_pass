import torch
from pass_dir.shared_kernels import shared_dispatch


# ---------- FX pattern / replacement ----------
# Matches: roll(3,3)+slice[:32,:32]+view+add  →  single output tmp_8
# Pattern inputs: (in_2=residual, in_3=attention_out)

def pattern(in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    tmp_8 = in_2 + tmp_7
    return tmp_8  # SINGLE output


def replacement_args(in_2, in_3):
    # a=in_2, b=in_3, c=in_2(dummy), route='r35_384'
    return (in_2, in_3, in_2, 'r35_384')


def replacement_func():
    return shared_dispatch