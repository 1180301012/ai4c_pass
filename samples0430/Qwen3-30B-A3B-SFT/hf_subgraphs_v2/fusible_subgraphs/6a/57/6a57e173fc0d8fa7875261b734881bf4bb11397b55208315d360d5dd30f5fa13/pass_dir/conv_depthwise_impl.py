import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SD': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SD': 512}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SD': 512}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SD': 512}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SD': 512}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SD': 512}, num_warps=4,  num_stages=3),
    ],
    key=['B', 'S', 'D', 'C'],
)
@triton.jit
def _depthwise_conv_add_permute_kernel(
    in2_ptr, in0_ptr, in1_ptr, out_ptr,
    B, S, D, C, K,
    BLOCK_SD: tl.constexpr,
):
    """
    Fused depthwise-conv + residual-add + permute kernel.

    Grid: (B * C,) — one program per (batch, channel) pair.
    Each program writes out[b, 0:S, c, 0:D] = conv(in2)[b,c,:,:] + in1[b,c,:,:]
    All in the [B, S, C, D] layout (i.e. permute(0,2,1,3) is implicit).
    """
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # ---- pre-load K filter weights for channel c -------------------------
    k_offs = tl.arange(0, BLOCK_SD)
    k_mask = k_offs < K
    w = tl.load(in0_ptr + c * K + k_offs, mask=k_mask, other=0.0)

    # ---- iterate over output positions (s, d) ----------------------------
    sd_offs = tl.arange(0, BLOCK_SD)
    sd_mask = sd_offs < S * D
    s_idx = sd_offs // D
    d_idx = sd_offs % D

    acc = tl.zeros([BLOCK_SD], dtype=tl.float32)

    # base pointer offsets (no s/d contribution yet)
    in2_base = b * (C * S * D) + c * (S * D)
    in1_base = b * (C * S * D) + c * (S * D)

    for k in range(K):
        h = s_idx + k - 32          # actual input row (may be out of range)
        valid = (h >= 0) & (h < S)

        in2_ptr_k = in2_base + h * D + d_idx
        x = tl.load(in2_ptr + in2_ptr_k, mask=sd_mask & valid, other=0.0)
        acc = acc + x * w[k]

    # ---- add residual ----------------------------------------------------
    res = tl.load(in1_ptr + in1_base + sd_offs, mask=sd_mask, other=0.0)
    result = acc + res

    # ---- store output in [B, S, C, D] layout -----------------------------
    # offset = b*(S*C*D) + s*(C*D) + c*D + d
    out_offs = b * (S * C * D) + s_idx * (C * D) + c * D + d_idx
    tl.store(out_ptr + out_offs, result.to(out_ptr.dtype.element_ty), mask=sd_mask)


@torch.fx.wrap
def depthwise_conv_add_permute(in_0, in_1, in_2):
    """
    Replacement for: conv2d + iadd + permute(0,2,1,3) + contiguous + view.

    in_0 : [C, 1, K, 1]  conv weights (depthwise)
    in_1 : [B, C, S, D]  residual
    in_2 : [B, C, S, D]  input to conv
    returns: [B, S, C*D]
    """
    B = in_1.shape[0]
    C = in_1.shape[1]
    S = in_1.shape[2]
    D = in_1.shape[3]
    K = in_0.shape[2]   # kernel height (=65)

    out = torch.empty(B, S, C, D, dtype=in_1.dtype, device=in_1.device)

    _depthwise_conv_add_permute_kernel[(B * C,)](
        in_2, in_0, in_1, out,
        B, S, D, C, K,
    )

    # view is zero-copy — fuse it so FX graph has a single call
    return out.view(B, S, C * D)