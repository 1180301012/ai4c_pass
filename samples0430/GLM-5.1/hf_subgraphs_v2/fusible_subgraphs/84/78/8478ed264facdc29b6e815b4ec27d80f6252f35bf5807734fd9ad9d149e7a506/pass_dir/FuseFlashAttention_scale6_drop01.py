import torch

# Pattern for flash attention with scale=6.0 and dropout(p=0.1, False, False)
# Matches hf-tiny-model-private_tiny-random-MobileViTModel float16 variant
# Note: dropout with training=False is identity regardless of p
def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul / 6.0
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul_1 = torch.matmul(tmp_3, in_2)
    tmp_5 = matmul_1.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return (tmp_6,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_scale6_drop01")

from pass_dir.flash_attn_kernel import flash_attn_dispatch

def replacement_func():
    return flash_attn_dispatch