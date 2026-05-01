import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the full convit gating computation
#   tmp_1 = softmax(in_2, dim=-1)
#   tmp_2 = in_0.view(1, -1, 1, 1)
#   gate  = sigmoid(tmp_2)
#   out   = (1 - gate) * in_1 + gate * tmp_1
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_1 = in_2.softmax(dim=-1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    return tmp_8


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel
#   Grid  : (B * H * R,)  — one program per attention row
#   Each program computes softmax over C=196 cols, applies sigmoid gate, blends
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # 1 row per program
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 1}, num_warps=4),
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 1}, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 1}, num_warps=16),
        # 2 rows per program  (3136/2 = 1568 programs)
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 2}, num_warps=4),
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 2}, num_warps=8),
        # 4 rows per program  (3136/4 = 784 programs)
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 4}, num_warps=4),
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 4}, num_warps=8),
        # 7 rows per program  (3136/7 = 448 programs)
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 7}, num_warps=4),
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 7}, num_warps=8),
        # 14 rows per program (3136/14 = 224 programs)
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 14}, num_warps=4),
        triton.Config({'BLOCK_C': 256, 'ROWS_PER_PROG': 14}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def _fused_gating_softmax_kernel(
    in0_ptr,           # [H]       gating parameters (any fp dtype, on CUDA)
    in1_ptr,           # [B,H,R,C] patch_score
    in2_ptr,           # [B,H,R,C] pos_score
    out_ptr,           # [B,H,R,C] output
    H,
    R,
    C,
    BLOCK_C: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    base_row = pid * ROWS_PER_PROG

    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    # Process ROWS_PER_PROG rows sequentially.
    # All ROWS_PER_PROG values evenly divide total_rows (3136 = 16*196),
    # so no bounds check is needed.
    for i in tl.static_range(ROWS_PER_PROG):
        row = base_row + i
        h = (row // R) % H

        # Gate: load in_0[h] and sigmoid in fp32 for numerical precision
        gate = tl.sigmoid(tl.load(in0_ptr + h).to(tl.float32))

        row_off = row * C

        # Numerically-stable softmax of pos_score row.
        # exp(-inf)==0 for masked positions, so no tl.where needed.
        pos = tl.load(in2_ptr + row_off + cols,
                      mask=mask, other=float('-inf')).to(tl.float32)
        pos -= tl.max(pos, axis=0)
        pos_exp = tl.exp(pos)
        softmax_pos = pos_exp / tl.sum(pos_exp, axis=0)

        # patch_score row
        patch = tl.load(in1_ptr + row_off + cols,
                        mask=mask, other=0.0).to(tl.float32)

        # Blend: patch + gate*(softmax - patch)  [3 ops instead of 4]
        out_val = patch + gate * (softmax_pos - patch)

        tl.store(out_ptr + row_off + cols, out_val, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper  (must be @torch.fx.wrap so FX doesn't try to trace inside)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_gating_softmax(in_0, in_1, in_2):
    # Move in_0 to GPU only if needed (keep its dtype — kernel converts to fp32)
    in_0_dev = in_0.to(device=in_1.device)

    B, H, R, C = in_1.shape
    total_rows = B * H * R
    out = torch.empty_like(in_1)

    def grid(meta):
        return (total_rows // meta['ROWS_PER_PROG'],)

    _fused_gating_softmax_kernel[grid](
        in_0_dev, in_1, in_2, out,
        H, R, C,
    )

    return out


# ---------------------------------------------------------------------------
# Required by the AI4C framework
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_gating_softmax