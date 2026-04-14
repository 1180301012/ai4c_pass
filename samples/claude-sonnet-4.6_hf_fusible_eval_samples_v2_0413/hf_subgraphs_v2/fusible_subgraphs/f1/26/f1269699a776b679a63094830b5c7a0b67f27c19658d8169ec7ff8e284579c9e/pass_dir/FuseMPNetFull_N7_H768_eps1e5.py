import torch
import triton
import triton.language as tl

# ===================== Triton Kernel =====================
# Reuse the H=768 kernel by importing from the N11 pass

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 1024}, num_warps=4),
        triton.Config({'BLOCK_H': 1024}, num_warps=8),
        triton.Config({'BLOCK_H': 512},  num_warps=8),
        triton.Config({'BLOCK_H': 1024}, num_warps=16),
    ],
    key=['H'],
)
@triton.jit
def fused_emb_add_layernorm_kernel_n7_h768(
    idx1_ptr,
    idx2_ptr,
    emb1_ptr,
    emb2_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    idx1 = tl.load(idx1_ptr + pid)
    idx2 = tl.load(idx2_ptr + pid)

    h_offs = tl.arange(0, BLOCK_H)
    mask = h_offs < H

    emb1 = tl.load(emb1_ptr + idx1 * H + h_offs, mask=mask, other=0.0)
    emb2 = tl.load(emb2_ptr + idx2 * H + h_offs, mask=mask, other=0.0)
    orig_dtype = emb1.dtype

    x = emb1.to(tl.float32) + emb2.to(tl.float32)

    # Layer norm
    mean = tl.sum(x, axis=0) / H
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * rstd

    w = tl.load(weight_ptr + h_offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + h_offs, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b

    tl.store(out_ptr + pid * H + h_offs, out.to(orig_dtype), mask=mask)


# ===================== Wrapper =====================

@torch.fx.wrap
def fused_emb_layernorm_N7_H768_eps1e5(in_0, in_1, in_2, in_3, in_4, in_5):
    H = 768
    eps = 1e-05

    device = in_5.device   # position_ids is always on CUDA
    dtype = in_4.dtype

    emb1 = in_4.to(device)
    emb2 = in_3.to(device)
    weight = in_2.to(device)
    bias = in_1.to(device)
    idx1 = in_0.to(device).reshape(-1)
    idx2 = in_5.reshape(-1)

    BS = idx1.shape[0]
    out = torch.empty(BS, H, dtype=dtype, device=device)

    fused_emb_add_layernorm_kernel_n7_h768[(BS,)](
        idx1, idx2, emb1, emb2, weight, bias, out,
        H=H, eps=eps,
    )

    B_orig, S_orig = in_0.shape
    return out.reshape(B_orig, S_orig, H)


# ===================== Pattern / Replacement =====================

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (768,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_emb_layernorm_N7_H768_eps1e5