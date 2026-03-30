import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Compatibility: torch.sym_sum is not available in older PyTorch builds.
# Patch it as a simple summation so models that reference it can be traced
# and executed without crashing.
# ---------------------------------------------------------------------------
if not hasattr(torch, 'sym_sum'):
    def _sym_sum(args):
        """Fallback implementation of torch.sym_sum."""
        result = args[0]
        for a in args[1:]:
            result = result + a
        return result
    torch.sym_sum = _sym_sum


# ---------------------------------------------------------------------------
# Pattern: match ONLY the spatial mean over dims (2, 3) with keepdim=True.
#
# The Dynamo FX graph keeps operations at the Python level (not ATen),
# preserving method calls as call_method nodes with original kwargs.
# We match ONLY the mean (not relu) to avoid arg-normalization mismatches
# on the relu inplace=True kwarg caused by ForceArgsTracer.
#
# The relu output (tmp_0) is observable outside the subgraph but is an
# INPUT (placeholder) to the matched subgraph, not an intermediate —
# so it need not appear in pattern outputs.
# ---------------------------------------------------------------------------
def pattern(x):
    tmp_3 = x.mean((2, 3), keepdim=True)
    return tmp_3


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: spatial mean over H*W (dims 2,3) with keepdim=True
#
# Grid  : (N*C,)  – one program per (batch, channel) slice
# Single-pass approach: load all HW elements at once (masked), reduce, store.
# BLOCK_HW must be >= HW; autotune picks the right size per HW value.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def spatial_mean_kernel(
    inp_ptr,
    out_ptr,
    HW,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    nc_id   = tl.program_id(0)
    base    = nc_id * HW
    offsets = tl.arange(0, BLOCK_HW)
    mask    = offsets < HW

    if IS_BF16:
        x = tl.load(inp_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    elif IS_FP16:
        x = tl.load(inp_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(inp_ptr + base + offsets, mask=mask, other=0.0)

    mean_val = tl.sum(x, axis=0) / HW

    if IS_BF16:
        tl.store(out_ptr + nc_id, mean_val.to(tl.bfloat16))
    elif IS_FP16:
        tl.store(out_ptr + nc_id, mean_val.to(tl.float16))
    else:
        tl.store(out_ptr + nc_id, mean_val)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX doesn't trace into it)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_spatial_mean(x):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    # Output: [N, C, 1, 1] matches keepdim=True mean over dims (2, 3)
    out = torch.empty((N, C, 1, 1), dtype=x.dtype, device=x.device)

    is_bf16 = x.dtype == torch.bfloat16
    is_fp16 = x.dtype == torch.float16

    spatial_mean_kernel[(NC,)](
        inp_ptr = x,
        out_ptr = out,
        HW      = HW,
        IS_BF16 = is_bf16,
        IS_FP16 = is_fp16,
    )

    return out


# ---------------------------------------------------------------------------
# Replacement entry point – return the callable, do NOT call it
# ---------------------------------------------------------------------------
def replacement_func():
    return triton_spatial_mean