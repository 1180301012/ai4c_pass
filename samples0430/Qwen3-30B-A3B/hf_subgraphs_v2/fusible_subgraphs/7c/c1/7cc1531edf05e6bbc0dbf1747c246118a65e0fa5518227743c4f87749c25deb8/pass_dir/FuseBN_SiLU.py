import torch
import triton
import triton.language as tl

@triton.jit
def bn_silu_kernel(
    x_ptr,
    rm_ptr,
    rv_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    start_idx = tl.program_id(0) * BLOCK_SIZE
    idx = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = idx < C * H * W

    # Calculate channel index per element
    channel = idx // (H * W)
    # Load BN parameters
    rm = tl.load(rm_ptr + channel, mask=channel < C, other=0.0)
    rv = tl.load(rv_ptr + channel, mask=channel < C, other=0.0)
    w = tl.load(w_ptr + channel, mask=channel < C, other=0.0)
    b = tl.load(b_ptr + channel, mask=channel < C, other=0.0)
    
    # Load input value
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    
    # BatchNorm: (x - rm) / sqrt(rv + eps) * weight + bias
    eps = 1e-05
    denom = tl.sqrt(rv + eps)
    bn = (x - rm) / denom
    bn = bn * w + b
    
    # SiLU activation: bn * sigmoid(bn)
    out = bn * tl.sigmoid(bn)
    
    tl.store(out_ptr + idx, out, mask=mask)

@torch.fx.wrap
def bn_silu(x, rm, rv, w, b):
    B, C, H, W = x.shape
    N = B * C * H * W
    BLOCK_SIZE = 256
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    bn_silu_kernel[(num_blocks,)](
        x_ptr=x,
        rm_ptr=rm,
        rv_ptr=rv,
        w_ptr=w,
        b_ptr=b,
        out_ptr=out,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def pattern(tmp_5, in_0, in_1, in_3, in_2):
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7

def replacement_args(tmp_5, in_0, in_1, in_3, in_2):
    return (tmp_5, in_0, in_1, in_3, in_2)

def replacement_func():
    return bn_silu