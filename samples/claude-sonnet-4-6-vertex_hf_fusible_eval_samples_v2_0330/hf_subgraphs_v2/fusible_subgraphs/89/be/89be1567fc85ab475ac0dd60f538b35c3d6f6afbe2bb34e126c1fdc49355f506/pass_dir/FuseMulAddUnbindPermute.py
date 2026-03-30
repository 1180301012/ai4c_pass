import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    return (tmp_6, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ------------------------------------------------------------------ #
# Triton kernel: fused mul-add for both j-slices, fully coalesced.   #
# BLOCK_C = C = 128 → no masking needed.                             #
# ------------------------------------------------------------------ #
@triton.jit
def _fused_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr,
    out_j0_ptr, out_j1_ptr,
    BK,
    BLOCK_C: tl.constexpr,
):
    pid    = tl.program_id(0)
    c_offs = tl.arange(0, BLOCK_C)
    row    = pid * BLOCK_C

    in_2   = tl.load(in_2_ptr + row + c_offs)
    in_1_0 = tl.load(in_1_ptr +           c_offs)
    in_1_1 = tl.load(in_1_ptr + BLOCK_C + c_offs)
    in_0_0 = tl.load(in_0_ptr +           c_offs)
    in_0_1 = tl.load(in_0_ptr + BLOCK_C + c_offs)

    tl.store(out_j0_ptr + row + c_offs, in_2 * in_1_0 + in_0_0)
    tl.store(out_j1_ptr + row + c_offs, in_2 * in_1_1 + in_0_1)


# ------------------------------------------------------------------ #
# Replacement: fully FX-traceable, no graph-breaks.                  #
#                                                                      #
# Uses the SAME broadcast computation as the original pattern but     #
# replaces  torch.unbind(dim=2)  with simple slice indexing.          #
# This gives torch.compile a cleaner IR to fuse:                     #
#   • mul + add → typically one fused elementwise kernel              #
#   • [:,:,j] indexing → pure stride adjustments (no GPU work)        #
#   • permute → stride adjustment (no GPU work)                       #
# Total: ~5 FX nodes vs original 6 (unbind replaced by getitem×2).   #
# ------------------------------------------------------------------ #
def _replacement_impl(in_0, in_1, in_2):
    """
    FX nodes:
      in_2.mul(in_1)       →  tmp   [B,K,2,C]   (1 GPU kernel, new tensor)
      tmp.add_(in_0)       →  tmp   [B,K,2,C]   (in-place: no extra allocation)
      tmp[:,:,0]           →  j0    [B,K,C]     (view)
      tmp[:,:,1]           →  j1    [B,K,C]     (view)
      j1.permute(0,2,1)    →  tmp_6 [B,C,K]     (view)

    In-place add_ saves one [B,K,2,C] tensor allocation vs the regular
    a*b+c pattern, reducing cudaMalloc pressure and memory bandwidth.
    2 returning nodes [tmp_6, j0] ↔ 2 pattern outputs ✓
    """
    tmp = in_2.mul(in_1)       # [B, K, 2, C]  broadcast mul → new contiguous tensor
    tmp.add_(in_0)             # in-place add of [2, C] bias (no extra allocation)
    out_j0 = tmp[:, :, 0]     # [B, K, C]  — view (select dim-2 = 0)
    out_j1 = tmp[:, :, 1]     # [B, K, C]  — view (select dim-2 = 1)
    tmp_6  = out_j1.permute(0, 2, 1)   # [B, C, K] — view
    return tmp_6, out_j0


def replacement_func():
    return _replacement_impl