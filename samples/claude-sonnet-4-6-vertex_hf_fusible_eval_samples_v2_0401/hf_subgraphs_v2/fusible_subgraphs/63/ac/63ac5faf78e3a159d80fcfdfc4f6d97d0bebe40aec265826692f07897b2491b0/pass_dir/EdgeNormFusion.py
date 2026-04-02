import torch
import torch.fx.proxy as fx_proxy
import triton
import triton.language as tl
from torch import inf


@triton.jit
def full_edge_norm_kernel(
    deg_ptr,
    row_ptr,
    col_ptr,
    edge_weight_ptr,
    out_ptr,
    n_edges,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully fused edge normalization from raw degree.
    Replaces all 7 GPU ops: pow_, __eq__, masked_fill_, getitem×2, mul×2.
    out[i] = rsqrt(deg[row[i]]) * edge_weight[i] * rsqrt(deg[col[i]])
    where rsqrt(0) = 0 (handles the deg=0 → inf → 0 masking).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_edges

    row_idx = tl.load(row_ptr + offsets, mask=mask, other=0)
    col_idx = tl.load(col_ptr + offsets, mask=mask, other=0)

    deg_row = tl.load(deg_ptr + row_idx, mask=mask, other=0.0).to(tl.float32)
    deg_col = tl.load(deg_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
    ew = tl.load(edge_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    norm_row = tl.where(deg_row > 0.0, tl.math.rsqrt(deg_row), 0.0)
    norm_col = tl.where(deg_col > 0.0, tl.math.rsqrt(deg_col), 0.0)

    out_f32 = norm_row * ew * norm_col

    if IS_BF16:
        out = out_f32.to(tl.bfloat16)
    else:
        out = out_f32.to(tl.float16)

    tl.store(out_ptr + offsets, out, mask=mask)


_BLOCK_SIZE = 256

# Pre-warm both IS_BF16 Triton kernel specializations at module load time.
# Using separate try blocks so float16 is compiled even if bfloat16 fails.
try:
    _pw_i = torch.zeros(4, dtype=torch.int64, device='cuda')
    _pw_d_bf = torch.ones(4, dtype=torch.bfloat16, device='cuda')
    _pw_o_bf = torch.empty(4, dtype=torch.bfloat16, device='cuda')
    full_edge_norm_kernel[(1,)](
        _pw_d_bf, _pw_i, _pw_i, _pw_d_bf, _pw_o_bf, 4, True,
        BLOCK_SIZE=256, num_warps=4,
    )
    del _pw_i, _pw_d_bf, _pw_o_bf
except Exception:
    pass  # CUDA not available; bfloat16 JIT will happen on first use

try:
    _pw_i2 = torch.zeros(4, dtype=torch.int64, device='cuda')
    _pw_d_f16 = torch.ones(4, dtype=torch.float16, device='cuda')
    _pw_o_f16 = torch.empty(4, dtype=torch.float16, device='cuda')
    full_edge_norm_kernel[(1,)](
        _pw_d_f16, _pw_i2, _pw_i2, _pw_d_f16, _pw_o_f16, 4, False,
        BLOCK_SIZE=256, num_warps=4,
    )
    del _pw_i2, _pw_d_f16, _pw_o_f16
except Exception:
    pass  # CUDA not available; float16 JIT will happen on first use


@torch.fx.wrap
def _compute_tmp8(in_3, in_5, in_4, in_2):
    """
    Fused Triton kernel for edge normalization (returns tmp_8).
    Takes raw degree in_3 and computes rsqrt(deg[row])*ew*rsqrt(deg[col]).
    """
    n_edges = in_4.numel()
    out = torch.empty_like(in_4)
    IS_BF16 = in_4.dtype == torch.bfloat16
    num_programs = (n_edges + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    full_edge_norm_kernel[(num_programs,)](
        in_3, in_5, in_2, in_4, out, n_edges, IS_BF16,
        BLOCK_SIZE=_BLOCK_SIZE, num_warps=4,
    )
    return out


def full_edge_norm_replacement(in_3, in_5, in_4, in_2):
    """
    Non-wrapped replacement producing TWO FX nodes (tmp_8 and tmp_4)
    to match the pattern's two returning anchors.
    The second anchor (tmp_4) re-uses the in_3 placeholder directly —
    no extra function call needed since tmp_4 has no external users.
    """
    tmp_8 = _compute_tmp8(in_3, in_5, in_4, in_2)
    # in_3 (placeholder node) serves as tmp_4 replacement.
    # tmp_4 has NO external users in the model, so its value doesn't matter.
    return tmp_8, in_3


def pattern(in_3, in_5, in_4, in_2):
    """
    Match the full edge normalization chain (all 7 GPU ops).

    Key tricks:
    1. fx_proxy.Attribute(tmp_2, '__eq__')(inf) creates call_method '__eq__'
       matching TorchDynamo's representation (not call_function torch.eq).
    2. Return (tmp_8, tmp_4): 2 anchors matching the replacement's 2 outputs.
       Also gives tmp_4 a user, preventing dead-code error.
    3. tmp_5, tmp_7 use tmp_2 (NOT tmp_4) — matches target (getitem uses pow_ output).
    4. All pow_ users (eq, masked_fill_, getitem×2) are in the match → no NOT_CONTAINED.
    """
    tmp_2 = in_3.pow_(-0.5)
    # Create call_method '__eq__' using Attribute, bypassing Proxy.__eq__ override
    # which would create call_function torch.eq instead.
    if isinstance(tmp_2, torch.fx.Proxy):
        tmp_3 = fx_proxy.Attribute(tmp_2, '__eq__')(inf)
    else:
        tmp_3 = (tmp_2 == inf)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    return tmp_8, tmp_4


def replacement_args(in_3, in_5, in_4, in_2):
    return (in_3, in_5, in_4, in_2)


def replacement_func():
    return full_edge_norm_replacement