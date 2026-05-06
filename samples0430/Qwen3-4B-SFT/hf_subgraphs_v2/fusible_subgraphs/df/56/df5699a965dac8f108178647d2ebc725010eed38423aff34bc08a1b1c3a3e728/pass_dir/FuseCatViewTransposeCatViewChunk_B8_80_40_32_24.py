"""Fuse: cat([in_3,tmp_4],1) -> view(8,2,40,32,24) -> transpose -> contiguous -> view(8,80,32,24) -> chunk(2). B=8, C=40, HW=768."""
import torch
from pass_dir.shuffle_kernel import shuffler_C40_HW768
def pattern(in_3, tmp_4):
    tmp_6  = torch.cat([in_3, tmp_4], dim=1)
    tmp_11 = tmp_6.view(8, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(8, 80, 32, 24)
    chunk_1 = tmp_14.chunk(2, dim=1)
    tmp_19 = chunk_1[0]
    tmp_20 = chunk_1[1]
    return tmp_19, tmp_20
def replacement_args(in_3, tmp_4): return (in_3, tmp_4)
@torch.fx.wrap
def _fused_b8_c40(in_3, tmp_4): out0, out1 = shuffler_C40_HW768(in_3, tmp_4); return out0, out1
def replacement_func(): return _fused_b8_c40