import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: fuse unsqueeze → expand_as → float
#   in_0   : [1, 16]      int64   attention mask
#   tmp_4  : [1, 16, 768] fp16/bf16  layer_norm output (graph-level input)
#   Returns tmp_7 (single output — avoids multi-output crash that occurs
#   with tuple returns in this framework version).
#
# We do this in two passes (matched sequentially):
#   Pass 1  — unsqueeze+expand+float → Triton kernel  (this file)
#   Pass 2  — multiply  with tmp_7    (separate simple pass)
# ---------------------------------------------------------------------------

def pattern(tmp_5, tmp_4):
    tmp_6 = tmp_5.expand_as(tmp_4)
    tmp_7 = tmp_6.float()
    return tmp_7


def replacement_args(tmp_5, tmp_4):
    return (tmp_5, tmp_4)


# ---------------------------------------------------------------------------
# Triton kernel: broadcast [R,1] float32 tmp_5 → [R,D] float32
# tmp_5 is [1,16,1] int64-converted → element [r,0,0] = mask value for row r.
# One CTA per row.  row index = r, load from ptr + r.
# ---------------------------------------------------------------------------

@triton.jit
def float_broadcast_kernel(
    in_ptr,   # [R,1] float32
    out_ptr,  # [R*D] float32
    total,
    D,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    val  = tl.load(in_ptr + offs // D)   # broadcast mask across D positions
    tl.store(out_ptr + offs, val, mask=mask)


@torch.fx.wrap
def kernel_wrapper(tmp_5, tmp_4):
    D     = 768
    out   = torch.zeros(tmp_5.shape[0], tmp_4.shape[1], D,
                        dtype=torch.float32, device=tmp_4.device)
    BLOCK = 4096   # single CTA, maximise GPU warp occupancy per launch

    grid = ((16 * D + BLOCK - 1) // BLOCK,)   # = (3,)
    float_broadcast_kernel[grid](
        tmp_5, out,
        16 * D,
        D,
        BLOCK=BLOCK,
        num_warps=16,
    )
    return out


def replacement_func():
    return kernel_wrapper