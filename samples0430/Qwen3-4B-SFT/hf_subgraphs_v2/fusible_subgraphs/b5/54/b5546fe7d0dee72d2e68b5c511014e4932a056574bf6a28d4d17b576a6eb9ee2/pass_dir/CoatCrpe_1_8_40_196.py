"""Pass: coat_crpe_1_8_40_196 N1=80 N2=120 C=120 H=14 W=14 wb=80 idx=16400"""
import torch
import triton
from pass_dir.coat_crpe_shared_kernel import coat_crpe_fused_kernel


def pattern(cat_out, in_4, in_6, scale):
    tmp_4 = cat_out.reshape(1, 8, 40, 196)
    tmp_5 = tmp_4.transpose(-1, -2)
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = scale * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_10.reshape(1, 197, 320)
    return tmp_11


def replacement_args(cat_out, in_4, in_6, scale): return (cat_out, in_4, in_6, scale)

@torch.fx.wrap
def _crpe_fused_1_8_40_196(cat_out, in_4, in_6, scale):
    H=W=14; C_conv=120; C_out=320; n=1*(H+1)*C_out   # 197*320=63040
    N1=cat_out.shape[1]-C_conv; N2=cat_out.shape[1]-2*C_conv-N1
    in_2, in_3, conv = cat_out[:, :N1], cat_out[:, N1:N1+N2], cat_out[:, 2*C_conv+N2:]
    output=torch.empty(1,H+1,C_out,dtype=in_4.dtype,device=in_4.device)
    grid=lambda m:((n+m['BLOCK_SIZE']-1)//m['BLOCK_SIZE'],)
    coat_crpe_fused_kernel[grid](in_2, in_3, conv, scale, in_4, in_6, output, N1, N2, C_conv, C_out, H, W, C_in2, n)
    return output

def replacement_func(): return _crpe_fused_1_8_40_196