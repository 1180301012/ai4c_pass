import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: fuse GELU + spatial mean reduction (dims 2,3 keepdim=True)
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton fused kernel – one block per (batch, channel) pair.
#
# Uses autotune keyed on (BC, HW) so that small-BC cases (low grid size,
# occupancy-limited) and large-BC cases (bandwidth-limited) each get the
# best BLOCK_SIZE / num_warps combination independently.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
    ],
    key=['BC', 'HW'],   # BC = B*C determines grid size; different optima for
                        # small vs large grids
)
@triton.jit
def fused_gelu_mean_kernel(
    input_ptr,
    gelu_output_ptr,
    mean_output_ptr,    # shape [BC] (= [B*C]) with stride 1; reshaped later
    BC, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid         = tl.program_id(0)   # 0 … BC-1
    base_offset = pid * HW

    # Load one scalar to pin the element dtype at compile time
    x_first = tl.load(input_ptr + base_offset)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask    = offsets < HW

        x     = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        # 1/sqrt(2) ≈ 0.70710678118
        gelu_x = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))

        tl.store(gelu_output_ptr + base_offset + offsets,
                 gelu_x.to(x_first.dtype), mask=mask)

        acc = acc + tl.where(mask, gelu_x, 0.0)

    # Reduce block-local accumulator → mean
    sum_val  = tl.sum(acc, axis=0)
    mean_val = (sum_val / HW).to(x_first.dtype)
    tl.store(mean_output_ptr + pid, mean_val)


# ---------------------------------------------------------------------------
# Kernel launcher – opaque to FX tracing via @torch.fx.wrap
# ---------------------------------------------------------------------------

@torch.fx.wrap
def gelu_mean_kernel_launcher(in_0):
    B, C, H, W = in_0.shape
    HW = H * W
    BC = B * C

    in_0_contig = in_0.contiguous()
    gelu_out    = torch.empty_like(in_0_contig)
    # mean_output is addressed as a flat [BC] array inside the kernel;
    # we allocate it as [B, C, 1, 1] so the view is free.
    mean_out    = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    fused_gelu_mean_kernel[(BC,)](
        in_0_contig, gelu_out, mean_out,
        BC, HW,
    )

    return gelu_out, mean_out


# ---------------------------------------------------------------------------
# Replacement function – NOT @torch.fx.wrap so FX traces through it.
# FX graph sees:
#   result = gelu_mean_kernel_launcher(in_0)   ← opaque call node
#   tmp_0  = result[0]                         ← getitem → matches pattern's tmp_0
#   tmp_1  = result[1]                         ← getitem → matches pattern's tmp_1
# Exactly 2 returning nodes → satisfies the pattern's output count.
# ---------------------------------------------------------------------------

def fused_gelu_mean(in_0):
    result = gelu_mean_kernel_launcher(in_0)
    tmp_0 = result[0]   # GELU output
    tmp_1 = result[1]   # spatial mean [B, C, 1, 1]
    return (tmp_0, tmp_1)


# ---------------------------------------------------------------------------
# replacement_func – must return the callable, not call it
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_gelu_mean