"""Fuse: cat([in_2,in_4],1) -> view(512,2,20,64,48) -> transpose -> contiguous -> view(512,40,64,48) -> chunk(2). B=512, C=20, HW=3072."""
import torch
from pass_dir.shuffle_kernel import shuffler_C20_HW3072
def pattern(in_2, in_4):
    tmp_5  = torch.cat([in_2, in_4], dim=1)
    tmp_7  = tmp_5.view(512, 2, 20, 64, 48)
    tmp_8  = torch.transpose(tmp_7, 1, 2)
    tmp_9  = tmp_8.contiguous()
    tmp_10 = tmp_9.view(512, 40, 64, 48)
    chunk  = tmp_10.chunk(2, dim=1)
    tmp_16 = chunk[0]
    tmp_17 = chunk[1]
    return tmp_16, tmp_17
def replacement_args(in_2, in_4): return (in_2, in_4)
@torch.fx.wrap
def _fused_b512_c20(in_2, in_4): out0, out1 = shuffler_C20_HW3072(in_2, in_4); return out0, out1
def replacement_func(): return _fused_b512_c20