import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_0 = torch.nn.functional.hardtanh(x, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    return tmp_1


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Fused hardtanh + global-avg-pool Triton kernel.
#
# Design goals (all shapes are tiny, so Python overhead dominates):
#  1. NO @triton.autotune → eliminates meta-dict lookup + lambda call overhead.
#  2. Pre-built tuple grid (no lambda) → fastest possible Triton launch.
#  3. Shape-keyed parameter cache → near-zero Python arithmetic per call.
#  4. BLOCK_HW capped at 256 → smaller kernel binary, lower register pressure,
#     faster Triton dispatch, and better SM occupancy.
#  5. NC_PER_BLOCK tuned so that ~64–80 CTAs are generated for small NC
#     and ~320–640 CTAs for large NC — minimises CTA-scheduling waves.
# ---------------------------------------------------------------------------
@triton.jit
def fused_hardtanh_avgpool_kernel(
    x_ptr,
    out_ptr,
    NC,
    HW,
    x_stride_nc,
    BLOCK_HW: tl.constexpr,
    NC_PER_BLOCK: tl.constexpr,
):
    """
    pid → CTA index.  Each CTA processes NC_PER_BLOCK (n,c) pairs
    sequentially (coalesced stride-1 HW loads), accumulates in fp32,
    stores with Triton's implicit type cast.
    """
    pid = tl.program_id(0)
    nc_base = pid * NC_PER_BLOCK

    for nc_i in range(NC_PER_BLOCK):
        nc = nc_base + nc_i
        if nc < NC:
            x_base = x_ptr + nc * x_stride_nc
            acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

            # Inner spatial loop — single iteration when BLOCK_HW >= HW
            for start in range(0, HW, BLOCK_HW):
                offsets = start + tl.arange(0, BLOCK_HW)
                mask = offsets < HW
                # Masked-out lanes get 0.0 → hardtanh(0)=0 → contribute 0
                x = tl.load(x_base + offsets, mask=mask, other=0.0).to(tl.float32)
                x = tl.minimum(tl.maximum(x, 0.0), 6.0)   # hardtanh [0, 6]
                acc = acc + x

            tl.store(out_ptr + nc, tl.sum(acc) / HW)


# ---------------------------------------------------------------------------
# Shape-keyed parameter cache (populated once, reused every call).
# ---------------------------------------------------------------------------
_params_cache: dict = {}


def _compute_params(shape):
    N, C, H, W = shape
    HW = H * W
    NC = N * C

    # BLOCK_HW: smallest power-of-2 >= HW, capped at 512 (single spatial pass).
    bh = 16
    while bh < HW:
        bh <<= 1
    BLOCK_HW = min(bh, 512)

    # NC_PER_BLOCK: grow until next doubling would leave < 128 CTAs, max 64.
    # Targets ~128-640 CTAs balancing scheduling waves vs launch overhead.
    nc_pb = 1
    while nc_pb < 64 and (NC // (nc_pb * 2)) >= 128:
        nc_pb <<= 1

    # num_warps = BLOCK_HW / 64: half the "standard" warp count.
    # Each hardware thread handles 2 SIMD elements, allowing 2x more CTAs
    # per SM.  For BLOCK_HW=512 this turns 1.43 scheduling waves into 0.71
    # (all CTAs run simultaneously) on the A30.
    num_warps = max(1, BLOCK_HW >> 6)   # BLOCK_HW / 64, minimum 1

    grid = (triton.cdiv(NC, nc_pb),)      # pre-built tuple — no lambda

    return BLOCK_HW, nc_pb, num_warps, grid, N, C, HW, NC


@torch.fx.wrap
def fused_hardtanh_avgpool(x):
    shape = x.shape
    entry = _params_cache.get(shape)
    if entry is None:
        entry = _compute_params(shape)
        _params_cache[shape] = entry

    BLOCK_HW, NC_PER_BLOCK, num_warps, grid, N, C, HW, NC = entry

    out = torch.empty(N, C, 1, 1, dtype=x.dtype, device=x.device)

    fused_hardtanh_avgpool_kernel[grid](
        x, out,
        NC, HW, HW,          # x_stride_nc = H*W (contiguous tensor)
        BLOCK_HW=BLOCK_HW,
        NC_PER_BLOCK=NC_PER_BLOCK,
        num_warps=num_warps,
    )

    return out


def replacement_func():
    return fused_hardtanh_avgpool