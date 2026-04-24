import torch
import triton
import triton.language as tl


@triton.jit
def _fused_attn_weighted_sum(
    w_ptr, v_ptr, out_ptr,
    D: tl.constexpr,
):
    """
    Weighted sum for scaled-dot-product attention result.
    w: [H, 1, 1] (attention weights, scalar per head)
    v: [H, 1, D] (value states)
    out: [1, 1, H*D] flattened output
    One Triton program per head.
    """
    pid = tl.program_id(0)
    offsets = tl.arange(0, D)
    w = tl.load(w_ptr + pid).to(tl.float32)
    v = tl.load(v_ptr + pid * D + offsets).to(tl.float32)
    out = (w * v).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + pid * D + offsets, out)


@torch.fx.wrap
def fused_attn_weighted_sum(in_0, in_1):
    """
    in_0: attention weights [H, 1, 1]  (output of softmax+dropout)
    in_1: value states      [H, 1, D]
    Returns: [1, 1, H*D] — the final attention output.
    """
    H = in_1.shape[0]
    D = in_1.shape[2]
    out = torch.empty((1, 1, H * D), dtype=in_1.dtype, device=in_1.device)
    if D == 32:
        _fused_attn_weighted_sum[(H,)](in_0, in_1, out, D=32, num_warps=1)
    elif D == 64:
        _fused_attn_weighted_sum[(H,)](in_0, in_1, out, D=64, num_warps=2)
    return out