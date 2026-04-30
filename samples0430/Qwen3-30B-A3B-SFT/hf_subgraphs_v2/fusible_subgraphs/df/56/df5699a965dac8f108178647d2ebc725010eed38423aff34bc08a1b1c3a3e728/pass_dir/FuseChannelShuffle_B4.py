import torch
import triton
import triton.language as tl
from pass_dir.channel_shuffle_kernels import run_channel_shuffle_stream1, run_channel_shuffle_stream2, run_sigmoid
from pass_dir.FuseChannelShuffle_B512 import _conv1x1_kernel, _mul_kernel


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = in_5 * tmp_3
    tmp_5 = torch.cat([in_2, in_4], dim=1)
    tmp_6 = torch.cat([in_3, tmp_4], dim=1)
    tmp_7  = tmp_5.view(4, 2, 20, 64, 48)
    tmp_8  = torch.transpose(tmp_7, 1, 2)
    tmp_9  = tmp_8.contiguous()
    tmp_10 = tmp_9.view(4, 40, 64, 48)
    tmp_11 = tmp_6.view(4, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(4, 80, 32, 24)
    chunk      = tmp_10.chunk(2, dim=1)
    tmp_16 = chunk[0]
    tmp_17 = chunk[1]
    chunk_1    = tmp_14.chunk(2, dim=1)
    tmp_19 = chunk_1[0]
    tmp_20 = chunk_1[1]
    return (tmp_16, tmp_19, tmp_17, tmp_20)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@torch.fx.wrap
def _fused_litehrnet_b4(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    B = 4
    tmp_16 = torch.empty_like(in_2)
    tmp_17 = torch.empty_like(in_2)
    run_channel_shuffle_stream1(in_2, in_4, tmp_16, tmp_17, B)
    conv_out_buf = torch.empty((B, 40, 1, 1), dtype=in_6.dtype, device=in_6.device)
    _conv1x1_kernel[triton.cdiv(B * 40, 256),](in_6, in_1, in_0, conv_out_buf, B, 40, 10, BLOCK=256,)
    sig_out   = torch.empty((B, 40, 1, 1), dtype=in_6.dtype, device=in_6.device)
    run_sigmoid(conv_out_buf, sig_out)
    tmp_4_out = torch.empty_like(in_5)
    n_mul     = in_5.numel()
    _mul_kernel[triton.cdiv(n_mul, 1024),](in_5, sig_out, tmp_4_out, n_mul, BLOCK=1024,)
    tmp_19 = torch.empty_like(in_3)
    tmp_20 = torch.empty_like(in_3)
    run_channel_shuffle_stream2(in_3, tmp_4_out, tmp_19, tmp_20, B)
    return (tmp_16, tmp_19, tmp_17, tmp_20)


def replacement_func():
    return _fused_litehrnet_b4