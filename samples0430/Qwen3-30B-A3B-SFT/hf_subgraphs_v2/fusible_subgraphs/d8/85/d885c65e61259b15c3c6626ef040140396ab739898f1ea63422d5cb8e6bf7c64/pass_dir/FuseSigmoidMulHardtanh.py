import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: sigmoid -> broadcast-multiply -> hardtanh (ReLU6)
#   conv_out  : [N, C, 1, 1]  – tiny SE-attention scale
#   in_2      : [N, C, H, W]  – large feature map
# ---------------------------------------------------------------------------
def pattern(conv_out, in_2):
    sig   = conv_out.sigmoid()
    tmp_4 = in_2 * sig
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(conv_out, in_2):
    return (conv_out, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fused sigmoid * in_2, then clamp to [0, 6]
#
#   1-D grid: dim-0 = NC  (one program per (n,c) pair)
#
#   Each program loops over all HW spatial elements in BLOCK_HW-sized tiles.
#   With `num_stages=3` (set at launch) Triton software-pipelines the loop:
#     load tile i+2  while computing tile i
#   This hides memory-load latency (typically 200-400 ns on A30 HBM2e)
#   and gives a significant bandwidth improvement over the single-tile grid.
#
#   For our shapes:
#     HW=784  (BLOCK_HW=1024): 1 iteration  → same as before
#     HW=1024 (BLOCK_HW=1024): 1 iteration  → same as before
#     HW=2304 (BLOCK_HW=1024): 3 iterations → pipelining gives real gains
# ---------------------------------------------------------------------------
@triton.jit
def _fused_sigmoid_mul_relu6_kernel(
    in2_ptr,
    conv_ptr,
    out_ptr,
    HW,                          # H*W  – runtime (enables dynamic loop bound)
    BLOCK_HW: tl.constexpr,     # tile size = 1024
):
    nc_id = tl.program_id(0)

    # Scalar sigmoid scale for this NC pair (broadcast to all HW lanes)
    scale = tl.load(conv_ptr + nc_id).to(tl.float32)
    scale = tl.sigmoid(scale)

    base = nc_id * HW

    # Dynamic loop over HW tiles – num_stages=3 pipelines loads/stores
    for hw_start in tl.range(0, HW, BLOCK_HW, num_stages=3):
        hw_offs = hw_start + tl.arange(0, BLOCK_HW)
        mask    = hw_offs < HW

        in2_val = tl.load(in2_ptr + base + hw_offs, mask=mask, other=0.0).to(tl.float32)
        result  = tl.minimum(tl.maximum(in2_val * scale, 0.0), 6.0)
        tl.store(out_ptr + base + hw_offs, result.to(in2_val.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX tracing treats it as a leaf node)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_sigmoid_mul_hardtanh(conv_out, in_2):
    N, C, H, W = in_2.shape
    out    = torch.empty_like(in_2)
    NC     = N * C
    HW     = H * W

    BLOCK_HW = 512
    # 1-D grid (one program per NC pair); loop inside kernel handles all HW
    # BLOCK_HW=512 gives:
    #   HW=784  → 2 iters, 76.6% eff  | HW=1024 → 2 iters, 100% eff
    #   HW=2304 → 5 iters, 92.2% eff (vs 75% with BLOCK_HW=1024, 3 iters)
    # num_stages=4 pipelines loads across 4 simultaneous in-flight iterations.
    _fused_sigmoid_mul_relu6_kernel[(NC,)](
        in_2, conv_out, out,
        HW=HW,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=4,   # pipeline 4 in-flight iterations → hides load latency
    )
    return out


# ---------------------------------------------------------------------------
def replacement_func():
    return fused_sigmoid_mul_hardtanh