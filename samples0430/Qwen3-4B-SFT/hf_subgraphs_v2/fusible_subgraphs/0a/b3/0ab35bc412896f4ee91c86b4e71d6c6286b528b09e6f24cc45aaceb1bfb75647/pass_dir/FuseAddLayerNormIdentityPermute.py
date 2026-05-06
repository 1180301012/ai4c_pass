import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: matches the entire subgraph in model.py exactly
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)
    tmp_4 = tmp_3.reshape(1, 2, 2, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.permute(0, 2, 3, 1)
    tmp_8 = tmp_7.reshape(1, -1, 128)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel – fused add + layernorm
#
#   Grid = (4,): each program owns one row of 128 elements.
#   H = BLOCK_H = 128: no masking, no wasted lanes.
#   num_warps=8, num_stages=2: empirically best on A30.
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def fused_add_layernorm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    H: tl.constexpr,      # = 128
    EPS: tl.constexpr,    # = 1e-5
    BLOCK_H: tl.constexpr,  # = 128
):
    row  = tl.program_id(0) * H
    offs = tl.arange(0, BLOCK_H)

    x = tl.load(x_ptr + row + offs).to(tl.float32)
    y = tl.load(y_ptr + row + offs).to(tl.float32)
    z = x + y

    mean = tl.sum(z, 0) / H
    dev  = z - mean
    var  = tl.sum(dev * dev, 0) / H
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(w_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)

    tl.store(out_ptr + row + offs, dev * rstd * w + b)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper  (must be @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_add_layernorm_wrapper(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [128]
    in_1 : weight [128]
    in_2 : [1, 4, 128]
    in_3 : [1, 4, 128]
    returns       [1, 4, 128]
    """
    out = torch.empty_like(in_2)
    fused_add_layernorm_kernel[(4,)](
        in_2, in_3, in_1, in_0, out,
        H=128,
        EPS=1e-05,
        BLOCK_H=128,
        num_warps=8,
        num_stages=2,
    )
    return out


def replacement_func():
    return fused_add_layernorm_wrapper