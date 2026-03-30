import torch
import triton
import triton.language as tl


# ── Triton kernel ──────────────────────────────────────────────────────────────

@triton.jit
def add_layernorm_kernel_16(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_D)
    mask    = offsets < D

    x = tl.load(x_ptr + row_idx * D + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row_idx * D + offsets, mask=mask, other=0.0).to(tl.float32)
    v = x + y

    mean = tl.sum(v, axis=0) / D
    v_c  = tl.where(mask, v - mean, 0.0)
    var  = tl.sum(v_c * v_c, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    v_n  = v_c * rstd

    w   = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b   = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)
    out = v_n * w + b

    tl.store(out_ptr + row_idx * D + offsets, out, mask=mask)


# ── PyTorch wrapper ────────────────────────────────────────────────────────────

@torch.fx.wrap
def triton_add_layernorm_16(in_0, in_1, in_2, in_3):
    """
    in_0 : [B, S, 16] addend 1
    in_1 : [B, S, 16] addend 2
    in_2 : [16]       layer-norm bias
    in_3 : [16]       layer-norm weight
    """
    B, S, D = in_0.shape
    out     = torch.empty_like(in_0)

    add_layernorm_kernel_16[(B * S,)](
        x_ptr      = in_0,
        y_ptr      = in_1,
        weight_ptr = in_3,
        bias_ptr   = in_2,
        out_ptr    = out,
        D          = D,
        eps        = 1e-5,
        BLOCK_D    = 16,
        num_warps  = 1,
    )
    return out


# ── Pattern & Replacement ──────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    tmp_13 = in_0 + in_1
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (16,), in_3, in_2, 1e-05)
    tmp_15 = torch.nn.functional.dropout(tmp_14, p=0.1, training=False)
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return triton_add_layernorm_16