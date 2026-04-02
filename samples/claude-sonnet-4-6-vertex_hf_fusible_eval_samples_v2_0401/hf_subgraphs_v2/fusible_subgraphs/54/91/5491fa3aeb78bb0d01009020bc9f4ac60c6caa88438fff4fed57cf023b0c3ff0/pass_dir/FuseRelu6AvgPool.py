import torch
import triton
import triton.language as tl
import functools


# ── Pattern / replacement metadata ─────────────────────────────────────────

def pattern(x):
    tmp_0 = torch.nn.functional.hardtanh(x, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    return tmp_1


def replacement_args(x):
    return (x,)


# ── Triton kernel (no @autotune – stable, single code path per call) ────────
# Config is selected by _get_launch_config() heuristic; the kernel is
# compiled lazily per (BLOCK_HW, PAIRS) pair and cached by Triton.
# After warmup all needed binaries are cached – no surprises during benchmark.

@triton.jit
def fused_relu6_avgpool_kernel(
    x_ptr,
    out_ptr,
    HW,
    BC,
    C,
    BLOCK_HW: tl.constexpr,
    PAIRS: tl.constexpr,
):
    b  = tl.program_id(0)   # batch index
    cg = tl.program_id(1)   # channel-group index

    b_C     = b * C          # pre-compute once per CTA
    c_start = cg * PAIRS
    inv_HW  = 1.0 / HW       # reciprocal avoids PAIRS divisions per pair

    for p in range(PAIRS):
        c    = c_start + p
        bc   = b_C + c
        base = bc * HW

        acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

        for i in range(0, HW, BLOCK_HW):
            idx  = i + tl.arange(0, BLOCK_HW)
            mask = idx < HW
            xv   = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
            xv   = tl.maximum(xv, 0.0)
            xv   = tl.minimum(xv, 6.0)
            acc += xv

        tl.store(out_ptr + bc, tl.sum(acc) * inv_HW)


# ── Heuristic config selection ──────────────────────────────────────────────

@functools.lru_cache(maxsize=64)
def _get_launch_config(HW: int, BC: int):
    """
    Returns (BLOCK_HW, PAIRS, num_warps, num_stages).

    Strategy:
      • BLOCK_HW = next power-of-2 ≥ HW, clamped to [32, 512].
        Handles all HW elements in a single inner-loop iteration (no loop).
      • PAIRS = largest power-of-2 in {1,2,4,8,16} such that
        total_CTAs = BC // PAIRS ≥ ~300.
        Targets ~320 CTAs: fits in a single GPU scheduling wave on A30
        (56 SMs × 32 concurrent CTAs/SM = 1792 max), minimising
        CTA-setup overhead without starving the GPU of parallelism.
      • num_warps: 1 for BLOCK_HW≤32, 2 for ≤128, 4 otherwise.
      • num_stages: 2 when HW > BLOCK_HW (multi-iteration), else 1.
    """
    # BLOCK_HW
    npow2 = 32
    while npow2 < HW:
        npow2 <<= 1
    BLOCK_HW = min(512, npow2)

    # PAIRS: largest power-of-2 ≤ 16 keeping total CTAs ≥ 500
    PAIRS = 1
    for p in (16, 8, 4, 2):
        if BC >= p * 500:
            PAIRS = p
            break

    # num_warps
    if BLOCK_HW <= 32:
        num_warps = 1
    elif BLOCK_HW <= 128:
        num_warps = 2
    else:
        num_warps = 4

    # num_stages
    num_stages = 2 if HW > BLOCK_HW else 1

    return BLOCK_HW, PAIRS, num_warps, num_stages


# ── Python wrapper ──────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_relu6_avgpool(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    HW = H * W
    BC = B * C

    BLOCK_HW, PAIRS, num_warps, num_stages = _get_launch_config(HW, BC)

    out = torch.empty(BC, dtype=x.dtype, device=x.device)

    fused_relu6_avgpool_kernel[(B, C // PAIRS)](
        x, out, HW=HW, BC=BC, C=C,
        BLOCK_HW=BLOCK_HW, PAIRS=PAIRS,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out.view(B, C, 1, 1)


def replacement_func():
    return fused_relu6_avgpool