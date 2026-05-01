import torch
import triton
import triton.language as tl

from pass_dir.shared_dispatch import _fused_attn_dispatch


# ---------------------------------------------------------------
# Full-chain pattern: all ops replaced by one Triton kernel.
# Pattern does NOT include dropout (trying to match graph where
# dropout(p=0, training=False) is const-folded away or absent).
# Shapes: in_0=[8,1,32], in_1=[8,32,1], in_2=[8,1,32] -> out=[1,1,256]
#
# Since T=1: softmax(bmm(q,k^T)) = 1.0, output = in_2 reshaped.
# Kernel just copies in_2 to output.
# ---------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    bmm = torch.bmm(in_0, in_1)
    tmp_1 = torch.nn.functional.softmax(bmm, dim=-1)
    bmm_1 = torch.bmm(tmp_1, in_2)
    tmp_4 = bmm_1.view(1, 8, 1, 32)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 256)
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    # Pass in_0 as first arg (unused in full route), in_2 as values
    return (in_0, in_2, "route_8h32d_full")


def replacement_func():
    return _fused_attn_dispatch