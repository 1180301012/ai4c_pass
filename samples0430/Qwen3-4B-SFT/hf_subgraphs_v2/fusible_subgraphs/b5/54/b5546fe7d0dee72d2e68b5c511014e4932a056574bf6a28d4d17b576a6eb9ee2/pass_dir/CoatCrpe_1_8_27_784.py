"""Pass: coat_crpe_1_8_27_784   N1=54 N2=81 C=81  H=28 W=28 wb=54 idx=13040"""
import torch
import triton
from pass_dir.coat_crpe_shared_kernel import coat_crpe_fused_kernel

def pattern(cat_out, in_4, in_6, scale):
    tmp_4=cat_out.reshape(1,8,27,784); tmp_5=tmp_4.transpose(-1,-2)
    tmp_6=in_6*tmp_5; tmp_7=torch.nn.functional.pad(tmp_6,(0,0,1,0,0,0),'constant',None)
    tmp_8=scale*in_4; tmp_9=tmp_8+tmp_7
    tmp_10=tmp_9.transpose(1,2); tmp_11=tmp_10.reshape(1,785,216); return tmp_11

def replacement_args(cat_out, in_4, in_6, scale): return (cat_out,in_4,in_6,scale)
@torch.fx.wrap def _f_1_8_27_784(cat_out,in_4,in_6,scale):
    H=W=28; C_conv=81; C_out=216; C_in2=54; n=216*785
    N1=cat_out.shape[1]-C_conv; N2=cat_out.shape[1]-2*C_conv-N1
    output=torch.empty(1,H+1,C_out,dtype=in_4.dtype,device=in_4.device)
    grid=lambda m:((n+m['BLOCK_SIZE']-1)//m['BLOCK_SIZE'],)
    coat_crpe_fused_kernel[grid](cat_out[:,:N1],cat_out[:,N1:N1+N2],cat_out[:,2*C_conv+N2:],scale,in_4,in_6,output,N1,N2,C_conv,C_out,H,W,C_in2,n)
    return output
def replacement_func():return _f_1_8_27_784