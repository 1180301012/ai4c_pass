import torch
import triton
import triton.language as tl


# ── Pattern to match (float32 graph, view-sizes 8×625×625) ────────────────────

def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 625, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 625, 625)
    tmp_4 = tmp_3.view(8, 625, 625)
    return (tmp_4, tmp_3)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ── Triton kernel: fused add + row-softmax ─────────────────────────────────────
# Grid  : (B*R,) = (8*625,) = (5000,)  — one program per (head, query_row)
# in_0  : [1, 1, 625, 625]  →  element (0,0,r,s) lives at r*625 + s
# in_1  : [1, 8, 625, 625]  →  element (0,b,r,s) lives at row_idx*625 + s
#          where row_idx = b*625 + r
# out   : flat [5000*625]  (float32, converted to original dtype in wrapper)

@triton.jit
def _fused_add_softmax_kernel_625_625(
    in0_ptr,             # fp32 pointer, shape [1,1,625,625]
    in1_ptr,             # fp32 pointer, shape [1,8,625,625]
    out_ptr,             # fp32 pointer, flat [5000*625]
    BLOCK_S: tl.constexpr,   # compile-time constant >= 625; we use 1024
):
    # B=8, R=625, S=625  — hardcoded for this kernel
    row_idx = tl.program_id(0)          # 0 .. 4999
    r = row_idx % 625                   # which row inside in_0

    offsets = tl.arange(0, BLOCK_S)
    mask    = offsets < 625

    # Load and upcast to fp32 for numerical stability
    x0 = tl.load(in0_ptr + r * 625 + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in1_ptr + row_idx * 625 + offsets, mask=mask, other=0.0).to(tl.float32)

    x = x0 + x1
    # Mask out-of-range lanes with -inf so they don't affect max/sum
    x = tl.where(mask, x, float('-inf'))

    # Numerically-stable softmax
    row_max  = tl.max(x, axis=0)
    x        = x - row_max
    exp_x    = tl.exp(x)
    exp_x    = tl.where(mask, exp_x, 0.0)
    row_sum  = tl.sum(exp_x, axis=0)
    softmax  = exp_x / row_sum

    tl.store(out_ptr + row_idx * 625 + offsets, softmax, mask=mask)


# ── Python wrapper (must be decorated with @torch.fx.wrap) ────────────────────

@torch.fx.wrap
def fused_add_softmax_8_625_625(in_0, in_1):
    """
    Fused add + softmax for the pattern:
        in_0 : [1, 1, 625, 625]  (attention mask, broadcast over 8 heads)
        in_1 : [1, 8, 625, 625]  (attention logits)
    Returns (tmp_5, tmp_3) matching the original dropout-inclusive output:
        tmp_3 : [1, 8, 625, 625]
        tmp_5 : [8, 625, 625]    (dropout(p=0) is identity → same data as tmp_3)
    """
    B, R, S = 8, 625, 625
    dtype  = in_1.dtype
    device = in_1.device

    # Allocate output buffer in float32
    out_f32 = torch.empty(B * R * S, dtype=torch.float32, device=device)

    _fused_add_softmax_kernel_625_625[(B * R,)](
        in_0, in_1, out_f32,
        BLOCK_S=1024,
        num_warps=8,
    )

    # Convert back to original dtype and build the two output views
    out   = out_f32.to(dtype)      # flat [B*R*S]
    tmp_5 = out.view(B, R, S)      # [8, 625, 625]
    tmp_3 = out.view(1, B, R, S)   # [1, 8, 625, 625]
    return (tmp_5, tmp_3)


def replacement_func():
    return fused_add_softmax_8_625_625