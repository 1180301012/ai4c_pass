import torch
import triton
import triton.language as tl


def pattern(in_4, in_5):
    # Match only the element-wise multiply (sigmoid-weight * feature-map)
    # in_4: [B, C, 1, 1], in_5: [B, C, H, W]
    # The broadcast: in_5 * in_4 → out[B,C,H,W]
    return in_5 * in_4


def replacement_args(in_4, in_5):
    return (in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _broadcast_mul_kernel(
    x_ptr,         # [B*C*H*W] - in_5 contiguous
    y_ptr,         # [B*C]     - in_4 (contiguous, reshaped)
    out_ptr,
    n_elements,    # total B*C*H*W
    n_bc,          # B*C
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # y has shape [B*C]; the same y[b*c] is used for all H*W positions
    y = tl.load(y_ptr + (offsets // (n_elements // n_bc)) % n_bc, mask=mask, other=0.0)
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_broadcast_mul(in_4, in_5):
    # in_4: [B, C, 1, 1] - contiguous, so flat index bc = b*C+c is correct
    # in_5: [B, C, H, W]
    B, C, H, W = in_5.shape
    n_elements = in_5.numel()
    n_bc = B * C
    out = torch.empty_like(in_5)

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _broadcast_mul_kernel[grid](
        in_5, in_4, out,
        n_elements, n_bc,
    )

    return out


def replacement_func():
    return fused_broadcast_mul