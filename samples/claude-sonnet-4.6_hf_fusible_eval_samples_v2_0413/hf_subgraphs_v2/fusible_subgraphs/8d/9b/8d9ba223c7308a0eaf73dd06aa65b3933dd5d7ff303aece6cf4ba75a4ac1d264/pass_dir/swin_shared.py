"""
Shared Triton kernel + dispatch: fused pos-bias-scale + 3-way broadcast-add.
Pattern covers: 16*sigmoid_out + in_2 + 2*in_3 (all 3 additions in one pass).
Both pass files share fused_swin_dispatch → satisfies replacement_func_limit=1.
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────
# Fused kernel: out = in_2 + 16*sigmoid_out + 2*in_3
#   sigmoid_out : [H, N, N]        (pre-computed sigmoid * will be scaled)
#   in_2        : [B, H, N, N]     (attention scores)
#   in_3        : [B, N, N]        (mask, added twice ≡ 2×)
#   out         : [B, H, N, N]
#   Grid: (B, H)  –  each program handles N_N = N*N elements (one attention head-batch pair)
# ─────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['B', 'H'],
)
@triton.jit
def _fused_bias_add_kernel(
    sig_ptr, in2_ptr, in3_ptr, out_ptr,
    B, H,
    N_N: tl.constexpr,   # N*N = 4096
):
    batch = tl.program_id(0)   # in [0, B)
    head  = tl.program_id(1)   # in [0, H)

    offs = tl.arange(0, N_N)

    # in_2[batch, head, :, :] and out[batch, head, :, :]
    bh_off = (batch * H + head) * N_N + offs
    # sigmoid_out[head, :, :]
    sig_off = head * N_N + offs
    # in_3[batch, :, :]
    mask_off = batch * N_N + offs

    in2_v = tl.load(in2_ptr + bh_off)
    sig_v = tl.load(sig_ptr + sig_off)
    in3_v = tl.load(in3_ptr + mask_off)

    # Fused: in_2 + 16*sigmoid_out + 2*in_3
    out = in2_v + 16.0 * sig_v.to(tl.float32).to(in2_v.dtype) + in3_v + in3_v

    tl.store(out_ptr + bh_off, out)


# ─────────────────────────────────────────────────────────────────
# Shared dispatch (IDENTICAL in both pass files)
# sigmoid_out: [H, N, N], in_2: [B, H, N, N], in_3: [B, N, N]
# Returns: [B, H, N, N]  (then model's view/softmax run on this)
# ─────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_swin_dispatch(sigmoid_out, in_2, in_3, route):
    B   = in_2.shape[0]
    H   = in_2.shape[1]
    N_N = in_2.shape[2] * in_2.shape[3]
    out = torch.empty_like(in_2)
    grid = (B, H)
    if route == "12h":
        _fused_bias_add_kernel[grid](sigmoid_out, in_2, in_3, out, B, H, N_N=N_N)
    elif route == "24h":
        _fused_bias_add_kernel[grid](sigmoid_out, in_2, in_3, out, B, H, N_N=N_N)
    return out