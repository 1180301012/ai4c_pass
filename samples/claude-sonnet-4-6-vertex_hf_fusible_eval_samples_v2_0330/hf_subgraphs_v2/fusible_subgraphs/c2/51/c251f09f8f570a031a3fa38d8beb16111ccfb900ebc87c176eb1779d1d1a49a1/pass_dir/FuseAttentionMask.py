import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: int64 attention mask  →  float32  →  1 - x  →  x * -3.4e38
# ---------------------------------------------------------------------------
# This chain appears identically in every graph and currently spawns 3
# separate CUDA kernels.  Fusing them into one eliminates 2 kernel-launch
# round-trips.  Using a BLOCK size matched to N reduces wasted mask lanes.
# ---------------------------------------------------------------------------

def pattern(x):
    y = x.to(dtype=torch.float32)
    z = 1.0 - y
    return z * -3.4028234663852886e+38


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel – parameterised by BLOCK (a constexpr, Triton compiles one
# binary per distinct BLOCK value used at runtime).
# ---------------------------------------------------------------------------

@triton.jit
def _attn_mask_fuse_kernel(
    x_ptr,          # int64 input  (values 0 or 1)
    out_ptr,        # float32 output
    N,              # total number of elements
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x_i64 = tl.load(x_ptr + offs, mask=mask, other=0)
    x_f32 = x_i64.to(tl.float32)

    # (1.0 - x_f32) * -3.4028234663852886e+38
    out = (1.0 - x_f32) * -3.4028234663852886e38

    tl.store(out_ptr + offs, out, mask=mask)


# ---------------------------------------------------------------------------
# Python-level wrapper – picks BLOCK and num_warps to fit N tightly
# (avoids wasting >50% of lanes and launches the minimum number of programs)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_attn_mask(x):
    N   = x.numel()
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)

    # BLOCK=256 with 1 warp gives the best balance:
    # - For large N (2048, 8192): 8-32 programs → good GPU parallelism
    # - For small N (11, 64): 1 program, fast kernel launch
    BLOCK = 256
    NW    = 1
    num_programs = (N + BLOCK - 1) // BLOCK
    _attn_mask_fuse_kernel[(num_programs,)](
        x, out, N,
        BLOCK=BLOCK,
        num_warps=NW,
    )
    return out


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_attn_mask