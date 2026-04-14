"""
Pass: FuseExpandSubtract
Fuses: in_0.view(1,1,32,512), in_4.unsqueeze(2).expand(1,4096,32,512), then subtraction

in_0: [32, 512]          float16/bfloat16  (codewords)
in_4: [1, 4096, 512]     float16/bfloat16  (features)
out:  [1, 4096, 32, 512] float16/bfloat16

Uses shared_fused_kernel with route="es" so both pass files share the same
replacement_func(), satisfying the output_pass_replacement_func_limit.
"""

import torch
from pass_dir.shared_kernels import shared_fused_kernel


def pattern(in_0, in_4):
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_10 = tmp_8 - tmp_6
    return tmp_10


def replacement_args(in_0, in_4):
    # 6-arg wrapper: a=in_0, b=in_4, c,d,e=dummies, route="es"
    return (in_0, in_4, in_0, in_0, in_0, "es")


def replacement_func():
    return shared_fused_kernel