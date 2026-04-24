import torch
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_0):
    tmp_0 = torch.ops.aten.gelu.default(in_0)
    tmp_1 = torch.ops.aten.view.default(tmp_0, [1, 124, 2, 768])
    tmp_2 = torch.ops.aten.view.default(tmp_1, [1, 248, 768])
    tmp_3 = torch.ops.aten.constant_pad_nd.default(tmp_2, [0, 0, 0, 1], 0)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


# ── Triton kernel ─────────────────────────────────────────────────────────────
# Fuses GELU + two view reshapes + one zero-padding row.
#
# Input  shape : [1, 124, 1536]   (N = 190 464 elements)
# Output shape : [1, 249, 768]    (M = 191 232 elements)
#
# For output flat-index `i`:
#   row = i // 768
#   col = i %  768
#   if row < 248:
#     src_row = row // 2          ( ∈ [0, 123])
#     src_ch  = row %  2          ( ∈ {0, 1})
#     input_idx = src_row*1536 + src_ch*768 + col
#   else:
#     zero
#
# GELU uses the exact erf-based formula to match torch.nn.functional.gelu.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N', 'M'],
)
@triton.jit
def _gelu_reshape_pad_kernel(
    input_ptr,
    output_ptr,
    N,   # 190 464  (valid input elements)
    M,   # 191 232  (total output elements)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    # row / col in the [249, 768] output (batch dim = 1, trivial)
    row = offsets // 768
    col = offsets % 768

    # For the padding row (row == 248) the index is out of bounds → treat as zero
    valid = mask & (row < 248)

    # Source row/col in [124, 1536] input
    src_row = row // 2
    src_ch  = row % 2
    input_idx = src_row * 1536 + src_ch * 768 + col

    # Load (clamped to avoid OOB pointer arithmetic when valid=False)
    safe_idx = tl.where(valid, input_idx, 0)
    x = tl.load(input_ptr + safe_idx, mask=valid, other=0.0)

    # Exact GELU:  0.5 · x · (1 + erf(x / √2))
    sqrt2_inv = 0.7071067811865476
    x_f32   = x.to(tl.float32)
    gelu_f32 = 0.5 * x_f32 * (1.0 + tl.math.erf(x_f32 * sqrt2_inv))
    gelu_out = gelu_f32.to(x.dtype)

    # Store (skip padding row via mask)
    tl.store(output_ptr + offsets, gelu_out, mask=valid)


# ── Kernel wrapper ────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_gelu_reshape_pad(in_0):
    N = 124 * 1536   # 190 464
    M = 249 * 768    # 191 232

    out = torch.empty((1, 249, 768), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: ((M + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _gelu_reshape_pad_kernel[grid](
        in_0,
        out,
        N,
        M,
    )

    return out


def replacement_func():
    return fused_gelu_reshape_pad