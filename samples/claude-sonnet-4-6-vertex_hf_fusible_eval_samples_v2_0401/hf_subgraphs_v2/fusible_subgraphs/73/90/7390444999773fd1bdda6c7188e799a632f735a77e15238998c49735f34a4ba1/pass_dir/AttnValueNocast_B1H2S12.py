import torch
import triton
import triton.language as tl


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['B', 'H', 'S'])
@triton.jit
def _vt_B1H2S12(inp_ptr, out_ptr, B, H, S):
    pid = tl.program_id(0)
    s = pid % S; bh = pid // S; b = bh // H; h = bh % H
    d = tl.arange(0, 64)
    tl.store(out_ptr + b*H*S*64 + h*S*64 + s*64 + d, tl.load(inp_ptr + b*S*H*64 + s*H*64 + h*64 + d))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['B', 'H', 'S'])
@triton.jit
def _ot_B1H2S12(inp_ptr, out_ptr, B, H, S):
    pid = tl.program_id(0)
    s = pid % S; bh = pid // S; b = bh // H; h = bh % H
    d = tl.arange(0, 64)
    tl.store(out_ptr + b*S*H*64 + s*H*64 + h*64 + d, tl.load(inp_ptr + b*H*S*64 + h*S*64 + s*64 + d))


@torch.fx.wrap
def _fw_B1H2S12(in_0, in_1, in_2, in_3, in_4, in_5):
    B, H = 1, 2
    device, dtype = in_3.device, in_3.dtype
    w = in_1.to(device=device, dtype=dtype); b = in_0.to(device=device, dtype=dtype)
    v_lin = in_3 @ w.transpose(0, 1) + b
    S = in_3.shape[1]
    v = torch.empty(B, H, S, 64, dtype=dtype, device=device)
    _vt_B1H2S12[(B*H*S,)](v_lin.view(B, S, H, 64), v, B, H, S)
    scale = 0.125
    attn_w = (in_5 @ in_4.transpose(-2, -1)).mul_(scale) + in_2
    attn_w = attn_w.softmax(dim=-1)
    sdpa_out = attn_w @ v
    out = torch.empty(B, S, H*64, dtype=sdpa_out.dtype, device=device)
    _ot_B1H2S12[(B*H*S,)](sdpa_out, out, B, H, S)
    return out


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    sdpa = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask=in_2, dropout_p=0.0, is_causal=False)
    tmp_6 = sdpa.transpose(1, 2)
    tmp_7 = tmp_6.reshape(1, 12, 128)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return _fw_B1H2S12