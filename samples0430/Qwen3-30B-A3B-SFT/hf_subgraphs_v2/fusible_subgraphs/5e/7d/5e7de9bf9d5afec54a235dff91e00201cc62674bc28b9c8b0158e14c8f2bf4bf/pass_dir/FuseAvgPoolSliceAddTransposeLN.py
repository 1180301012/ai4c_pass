import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: slice(tmp_5,:124) + slice(x2,:124) + add + transpose + layer_norm
#
# avg_pool1d is excluded so it produces its natural [1,768,125] output and
# tmp_5 remains observable outside this subgraph (used by tmp_6 = tmp_5[:,:,:124]).
# ---------------------------------------------------------------------------
def pattern(x1p, x2, weight, bias):
    # x1p : [1, C, 125] – avg_pool1d output (observable outside)
    # x2  : [1, C, 124] – gelu(conv1d) output
    tmp_6 = x1p[(Ellipsis, slice(None, 124, None))]
    tmp_7 = x2[(Ellipsis, slice(None, 124, None))]
    tmp_8 = tmp_6 + tmp_7
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), weight, bias, 1e-05)
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.1, False, False)
    return tmp_11


def replacement_args(x1p, x2, weight, bias):
    return (x1p, x2, weight, bias)


# ---------------------------------------------------------------------------
# Triton kernel: fused (slice_pool[:,:124] + slice_gelu[:,:124]) + transpose + LN
#
# Inputs (NCHW, N=1):
#   x1p  : [1, C, S1p=125]  – avg_pool1d output  (slice [:,:124] → [1,C,124])
#   x2   : [1, C, S2=124]   – gelu(conv1d) output
#   out  : [1, 124, C]      – layer_norm result (transposed)
#
# Grid = (124,) – one program per output sequence position.
# ---------------------------------------------------------------------------
# Single fused kernel: slice + add + transpose + layer_norm
# Grid = (S_out=125,) – one program per output sequence position.
# ---------------------------------------------------------------------------
@triton.jit
def fused_slice_add_ln_kernel(
    x1p_ptr,      # [1, C, S1p]  – avg_pool1d output
    x2_ptr,       # [1, C, S2]   – gelu(conv1d) output
    out_ptr,      # [1, S_out, C]
    weight_ptr,   # [C]
    bias_ptr,     # [C]
    C,            # hidden dim = 768
    S1p,          # pool output seq len = 125
    S2,           # gelu output seq len = 124
    S_out,        # output seq len = S1p = 125
    eps,
    BLOCK_H: tl.constexpr,
):
    pid_s = tl.program_id(0)   # in [0, S_out) = [0, 125)

    offsets = tl.arange(0, BLOCK_H)
    mask = offsets < C           # C=768, mask last 256

    # Load x1p[:, :, pid_s]  (stride S1p=125 between channels)
    x1p_off = offsets * S1p + pid_s
    x1p_val = tl.load(x1p_ptr + x1p_off, mask=mask, other=0.0).to(tl.float32)

    # Load x2[:, :, pid_s]  (stride S2=124 between channels)
    x2_off = offsets * S2 + pid_s
    x2_val = tl.load(x2_ptr + x2_off, mask=mask, other=0.0).to(tl.float32)

    # Add
    x = x1p_val + x2_val

    # Layer norm over C=768
    mean = tl.sum(x, axis=0) / C
    diff = x - mean
    var  = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias   = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

    out = diff * rstd * weight + bias

    # Store to NHWC output [1, S_out, C]
    out_off = pid_s * C + offsets
    tl.store(out_ptr + out_off, out.to(x2_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_slice_add_ln(x1p, x2, weight, bias):
    # x1p : [1, C, 125] – avg_pool1d output
    # x2  : [1, C, 124] – gelu(conv1d) output
    N, C, S1p = x1p.shape   # S1p=125
    _N, _C, S2  = x2.shape  # S2=124
    S_out = S1p             # output [N, 125, C] = expected [N, 125, 768]

    out = torch.empty((N, S_out, C), dtype=x2.dtype, device=x2.device)

    fused_slice_add_ln_kernel[(N * S_out,)](
        x1p, x2, out, weight, bias,
        C, S1p, S2, S_out,
        1e-05,
        BLOCK_H=1024, num_warps=8,
    )

    return out


def replacement_func():
    return fused_slice_add_ln