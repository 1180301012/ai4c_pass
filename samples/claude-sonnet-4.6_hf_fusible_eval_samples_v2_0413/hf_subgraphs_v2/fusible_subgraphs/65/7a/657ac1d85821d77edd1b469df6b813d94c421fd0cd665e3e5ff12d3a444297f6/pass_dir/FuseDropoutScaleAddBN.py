import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fuses scale-multiply + residual-add in one memory pass.
# Produces tmp_10 = dropped_conv * gamma + residual.
# gamma has shape [C, 1, 1]; all other tensors are [N, C, H, W] in NCHW layout.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def _fused_scale_add_kernel(
    conv_ptr,
    gamma_ptr,
    residual_ptr,
    out_ptr,
    total_elements,
    HW,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Channel index for NCHW layout: c = (flat_idx // HW) % C
    c_idx = (offsets // HW) % C

    # Load in native dtype
    conv_raw  = tl.load(conv_ptr     + offsets, mask=mask, other=0.0)
    gamma_raw = tl.load(gamma_ptr    + c_idx,   mask=mask, other=1.0)
    res_raw   = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

    # Compute in float32 for accuracy, store back in native dtype
    result_f = conv_raw.to(tl.float32) * gamma_raw.to(tl.float32) + res_raw.to(tl.float32)
    tl.store(out_ptr + offsets, result_f.to(conv_raw.dtype), mask=mask)


@torch.fx.wrap
def fused_scale_add(
    dropped_conv,  # [N, C, H, W]  output of the dropout(p=0) no-op
    gamma,         # [C, 1, 1]     layer-scale weights
    residual,      # [N, C, H, W]  skip-connection input
):
    N, C, H, W = dropped_conv.shape
    HW = H * W
    total_elements = N * C * HW

    out = torch.empty_like(residual)
    grid = lambda meta: ((total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _fused_scale_add_kernel[grid](
        dropped_conv, gamma, residual, out,
        total_elements, HW, C,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern: matches the mul + add subgraph.
# batch_norm is intentionally left outside the pattern to avoid Dynamo
# lowering mismatches that would cause pass crashes.
# ---------------------------------------------------------------------------
def pattern(dropped_conv, gamma, residual):
    scaled = dropped_conv * gamma
    tmp_10 = residual + scaled
    return tmp_10


def replacement_args(dropped_conv, gamma, residual):
    return (dropped_conv, gamma, residual)


def replacement_func():
    return fused_scale_add