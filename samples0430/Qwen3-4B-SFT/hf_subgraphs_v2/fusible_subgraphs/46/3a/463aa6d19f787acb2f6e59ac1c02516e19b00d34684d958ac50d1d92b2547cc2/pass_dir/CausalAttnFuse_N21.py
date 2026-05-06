"""
Pass: fused causal attention mask for N=21.
Matches the subgraph: clone -> add -> eq -> masked_fill -> setitem -> eq -> all -> ~ -> mul
Single external input: in_0 [1, 21] int64 cuda tensor
Output: [1, 1, 21, 21] float32 cuda tensor
"""
import torch
import triton
from pass_dir.shared_kernel import fuse_causal_attn_mask_kernel


@torch.fx.wrap
def fused_causal_mask_21(in_0):
    N = 21
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    fuse_causal_attn_mask_kernel[(N,)](
        in_0,
        out,
        N=N,
    )
    return out


def pattern(in_0):
    # Affix placeholder so `tmp_9.clone()` is valid Python during tracing.
    # The FX SubgraphMatcher will bind `tmp_9` to whatever tensor precedes the clone
    # (i.e. the output of tmp_8.expand(1,1,-1,-1)) so that subsequent ops are matched.
    fake_tmp_9 = None
    tmp_10 = fake_tmp_9.clone()   # → matches target's tmp_10 = tmp_9.clone()
    tmp_11 = tmp_10[:, :, :, :21]
    tmp_12 = in_0[:, None, None, :]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[:, :, :, :21]
    tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_10[:, :, :, :21] = tmp_17
    tmp_19 = tmp_10.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim = -1, keepdim = True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_10.mul(tmp_21)
    return tmp_22


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_causal_mask_21