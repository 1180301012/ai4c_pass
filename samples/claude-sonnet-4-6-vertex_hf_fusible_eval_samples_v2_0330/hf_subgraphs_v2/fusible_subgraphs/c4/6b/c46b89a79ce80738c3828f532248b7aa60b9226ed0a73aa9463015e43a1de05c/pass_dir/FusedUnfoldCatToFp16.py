import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: permute→reshape→permute→reshape→cat→to(float16)
#
# The FX model graph uses torch.nn.functional.unfold as a leaf node.
# Tracing the pattern with F.unfold fails (C extension im2col doesn't
# dispatch __torch_function__ properly), so we exclude unfold from the
# pattern and accept its outputs as placeholder inputs.
#
#   tmp_0: [1, 442368,  9]  = unfold(in_1, kernel=(384,384), stride=(192,192))
#   tmp_3: [1, 442368, 25]  = unfold(in_2, kernel=(384,384), stride=(288,288))
#   in_0:  [1, 3, 384, 384]
# ---------------------------------------------------------------------------
def pattern(tmp_0, tmp_3, in_0):
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim = 0)
    tmp_7 = tmp_6.to(dtype = torch.float16)
    return tmp_7


def replacement_args(tmp_0, tmp_3, in_0):
    return (tmp_0, tmp_3, in_0)


# ---------------------------------------------------------------------------
# Triton kernel
#
# Fuses permute+reshape+cat+cast into a single memory pass.
#
# tmp_0 layout [1, 442368, 9]:  element[0, col, patch] at offset col*9 + patch
# tmp_3 layout [1, 442368, 25]: element[0, col, patch] at offset col*25 + patch
# in_0  layout [1, 3, 384, 384]: flat offset = c*HW + h*W + w = col
#
# Output [35, 3, 384, 384] flat:
#   b ∈ [0,  25): tmp3[0, col, b]   (col = c*HW + h*W + w)
#   b ∈ [25, 34): tmp0[0, col, b-25]
#   b == 34:      in0[0, c, h, w]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
        triton.Config({"BLOCK_SIZE": 8192}),
        triton.Config({"BLOCK_SIZE": 16384}),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_perm_reshape_cat_fp16_kernel(
    tmp0_ptr,   # [1, 442368,  9]
    tmp3_ptr,   # [1, 442368, 25]
    in0_ptr,    # [1, 3, 384, 384]
    out_ptr,    # [35, 3, 384, 384] float16 (flat)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < n_elements

    CHW = 3 * 384 * 384   # 442368
    HW  = 384 * 384        # 147456
    W   = 384

    b   = offsets // CHW
    c   = (offsets // HW) % 3
    h   = (offsets % HW) // W
    w   = offsets % W
    col = c * HW + h * W + w   # [0, 442367]

    # Region A: in_2 patches (b < 25)  →  tmp3[0, col, b]
    off_a = col * 25 + b

    # Region B: in_1 patches (25 ≤ b < 34)  →  tmp0[0, col, b-25]
    off_b = col * 9 + (b - 25)

    # Region C: in_0 (b == 34)
    off_c = col

    m_a = mask & (b < 25)
    m_b = mask & (b >= 25) & (b < 34)
    m_c = mask & (b >= 34)

    v_a = tl.load(tmp3_ptr + off_a, mask=m_a, other=0.0)
    v_b = tl.load(tmp0_ptr + off_b, mask=m_b, other=0.0)
    v_c = tl.load(in0_ptr  + off_c, mask=m_c, other=0.0)

    val = tl.where(b < 25, v_a, tl.where(b < 34, v_b, v_c))
    tl.store(out_ptr + offsets, val.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_perm_reshape_cat_fp16(tmp_0, tmp_3, in_0):
    N = 35 * 3 * 384 * 384   # 15 482 880
    out = torch.empty(N, dtype=torch.float16, device=in_0.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    fused_perm_reshape_cat_fp16_kernel[grid](
        tmp_0.contiguous(),
        tmp_3.contiguous(),
        in_0.contiguous(),
        out,
        N,
    )
    return out.reshape(35, 3, 384, 384)


def replacement_func():
    return fused_perm_reshape_cat_fp16