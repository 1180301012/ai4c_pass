import sys
import os
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# (1) Make this file importable as 'SiluMeanViewFused' by torch.compile/dynamo.
#     Add pass_dir to sys.path so importlib.import_module('SiluMeanViewFused')
#     can find this file.
# ---------------------------------------------------------------------------
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)


# ---------------------------------------------------------------------------
# (2) torch.sym_sum does not exist in this environment but the model graphs use
#     it as dead code (result immediately set to None).  Patch it here at import
#     time so that torch.fx.symbolic_trace can trace the models without raising
#     AttributeError.
# ---------------------------------------------------------------------------
if not hasattr(torch, 'sym_sum'):
    def _sym_sum_stub(args):
        """Dead-code stub for torch.sym_sum used in eca_resnet subgraphs."""
        try:
            return sum(int(a) if not isinstance(a, torch.Tensor) else a.item()
                       for a in args)
        except Exception:
            return 0
    torch.sym_sum = _sym_sum_stub


# ---------------------------------------------------------------------------
# Pattern: silu (inplace) -> mean over spatial dims (2,3)
# The downstream view(1,1,-1) stays in the graph (cheap metadata op).
# ---------------------------------------------------------------------------

def pattern(x):
    tmp_0 = torch.nn.functional.silu(x, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    return (tmp_0, tmp_1)


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Fused Triton kernel: SiLU + spatial mean, one programme per (B, C) slice.
#
# Grid  : (B * C,)
# Per-programme work:
#   - iterate over the H*W elements in blocks of BLOCK_SIZE
#   - compute SiLU in fp32, write back in original dtype
#   - accumulate partial sums for the mean
#   - after the loop, divide by HW and store the per-channel mean
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32},   num_warps=1),
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def silu_mean_fused_kernel(
    x_ptr,
    out_silu_ptr,
    out_mean_ptr,
    HW,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bc_idx = tl.program_id(0)
    base   = bc_idx * HW

    # Accumulator lives in fp32 regardless of input dtype
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over the H*W spatial elements in tiles of BLOCK_SIZE
    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask    = offsets < HW

        # Load (fill out-of-bounds with 0; SiLU(0)=0 so mean stays correct)
        x     = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        # SiLU: x * sigmoid(x)
        silu_f32 = x_f32 * tl.sigmoid(x_f32)

        # Write SiLU output back in original dtype
        tl.store(out_silu_ptr + base + offsets, silu_f32.to(DTYPE), mask=mask)

        # Accumulate (out-of-bounds slots hold SiLU(0)=0, safe to add)
        acc = acc + silu_f32

    # Reduce accumulator and store per-channel mean
    total    = tl.sum(acc, axis=0)
    mean_val = (total / HW).to(DTYPE)
    tl.store(out_mean_ptr + bc_idx, mean_val)


# Map from torch dtype to Triton dtype constant
_DTYPE_MAP = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


@torch._dynamo.allow_in_graph
@torch.fx.wrap
def silu_mean_fused(x):
    B, C, H, W = x.shape
    HW = H * W

    out_silu = torch.empty_like(x)
    # Flat mean buffer; will be reshaped to [B, C] to match tmp_1's shape
    out_mean_flat = torch.empty(B * C, dtype=x.dtype, device=x.device)

    tl_dtype = _DTYPE_MAP[x.dtype]

    silu_mean_fused_kernel[(B * C,)](
        x,
        out_silu,
        out_mean_flat,
        HW,
        DTYPE=tl_dtype,
    )

    # Return mean with shape [B, C] — matching what tmp_0.mean((2,3)) produces.
    # The downstream view(1, 1, -1) node in the graph will do the reshape.
    out_mean = out_mean_flat.view(B, C)
    return (out_silu, out_mean)


def replacement_func():
    return silu_mean_fused