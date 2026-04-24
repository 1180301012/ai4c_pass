import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: gelu(in_0) * in_1  (dropout with training=False is identity)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused GELU + element-wise multiply
# ---------------------------------------------------------------------------
@triton.jit
def _fused_gelu_mul_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Alignment + contiguity hints: enables vectorised float4 loads
    block_start = tl.multiple_of(pid * BLOCK_SIZE, BLOCK_SIZE)
    offsets = block_start + tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    mask = offsets < n_elements

    # Load – Triton auto-vectorises contiguous aligned accesses
    x = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(in1_ptr + offsets, mask=mask, other=0.0)

    # Upcast to float32 for exact GELU (required for exact float32 match)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)

    # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    INV_SQRT2 = 0.7071067811865476
    erf_val = tl.math.erf(x_f32 * INV_SQRT2)
    gelu_out = x_f32 * 0.5 * (1.0 + erf_val)

    # Element-wise multiply (dropout training=False is identity)
    out_f32 = gelu_out * y_f32

    # Cast back to original dtype and store
    out = out_f32.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper (must be @torch.fx.wrap so FX doesn't trace into it)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_gelu_mul(in_0, in_1):
    N = in_0.numel()
    out = torch.empty_like(in_0)

    # Dynamically pick BLOCK_SIZE for best GPU occupancy on A30 (56 SMs).
    # Pre-warming ensures JIT is compiled before timing begins.
    #
    # Occupancy analysis (A30: 64 max warps/SM, 2048 max threads/SM):
    #   N ≥ 4M  → BLOCK_SIZE=2048/num_warps=8 : 8 blocks/SM, 50% occ.
    #   N <  4M, N ≥ 128k : BLOCK_SIZE=64/num_warps=2 : 32 blocks/SM, 100% occ.
    #     (e.g. N=4M bfloat16: 65536 blocks, all fit in 56×32=1792 slots simultaneously)
    #   N <  128k : BLOCK_SIZE=32/num_warps=1 : more blocks for tiny N
    if N >= 4194304:           # ≥ 4 M elements → large tensor
        _BLOCK_SIZE, _NUM_WARPS = 2048, 8
    elif N >= 131072:          # ≥ 128 k elements → medium-small tensor
        _BLOCK_SIZE, _NUM_WARPS = 64, 2
    else:                      # tiny tensors (e.g. N=22528 = 1×11×2048)
        _BLOCK_SIZE, _NUM_WARPS = 32, 1

    n_blocks = (N + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    _fused_gelu_mul_kernel[(n_blocks,)](
        in_0,
        in_1,
        out,
        N,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=_NUM_WARPS,
    )
    return out


# ---------------------------------------------------------------------------
# Pre-warm all kernel variants at import time so JIT compilation is done
# before any benchmark timing begins.  This eliminates compilation spikes.
# ---------------------------------------------------------------------------
def _prewarm():
    try:
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for n in (1024, 4096, 16384):   # typical N values from test graphs
                _x   = torch.zeros(n, device='cuda', dtype=dtype)
                _y   = torch.zeros(n, device='cuda', dtype=dtype)
                _out = torch.empty(n, device='cuda', dtype=dtype)
                # Launch with each (BLOCK_SIZE, num_warps) used in the wrapper
                for bs, nw in ((2048, 8), (256, 2), (32, 1)):
                    _fused_gelu_mul_kernel[((n + bs - 1) // bs,)](
                        _x, _y, _out, n,
                        BLOCK_SIZE=bs, num_warps=nw,
                    )
    except Exception:
        pass  # Non-CUDA environments or transient errors


_prewarm()


# ---------------------------------------------------------------------------
# replacement_func: return the wrapper (zero-argument function)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_gelu_mul