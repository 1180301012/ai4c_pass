import torch
import triton
import triton.language as tl


def pattern(in_0, relu_x, pool_x):
    tmp_4 = pool_x - relu_x
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = relu_x + tmp_7
    return tmp_8


def replacement_args(in_0, relu_x, pool_x):
    return (in_0, relu_x, pool_x)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
    ],
    key=['HW'],
)
@triton.jit
def _fused_elems_kernel(
    relu_x_ptr,   # (B, C, H, W) relu output   – NCHW contiguous
    pool_x_ptr,   # (B, C, H, W) avg_pool output
    scale_ptr,    # (C,) channel-wise scale (in_0)
    out_ptr,      # (B, C, H, W) result
    C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D launch: axis-0 = (b * C + c), axis-1 = hw-tile
    bc_idx = tl.program_id(0)   # batch-channel index: 0..B*C-1
    hw_pid = tl.program_id(1)   # HW tile index

    c_idx = bc_idx % C          # channel – computed once per block

    hw_offs = hw_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hw_mask = hw_offs < HW

    base    = bc_idx * HW
    offsets = base + hw_offs

    relu_x = tl.load(relu_x_ptr + offsets, mask=hw_mask, other=0.0).to(tl.float32)
    pool_x = tl.load(pool_x_ptr + offsets, mask=hw_mask, other=0.0).to(tl.float32)
    scale  = tl.load(scale_ptr  + c_idx).to(tl.float32)    # scalar per block

    out = relu_x + scale * (pool_x - relu_x)

    tl.store(out_ptr + offsets, out.to(relu_x_ptr.dtype.element_ty), mask=hw_mask)


@torch.fx.wrap
def meta4d_fused(in_0, relu_x, pool_x):
    C = relu_x.shape[1]
    N = relu_x.numel()

    # Fast path: for small tensors the Python-wrapper overhead dominates
    if N <= 2500000:
        return relu_x + in_0.view(C, 1, 1) * (pool_x - relu_x)

    # Slow path: fused Triton kernel for large tensors
    s   = relu_x.shape
    HW  = s[2] * s[3]
    BC  = s[0] * C
    out = torch.empty_like(relu_x)

    # 2-D grid: (B*C  x  ceil(HW/BLOCK_SIZE))
    grid = lambda meta: (BC, (HW + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])

    _fused_elems_kernel[grid](
        relu_x, pool_x, in_0, out,
        C, HW,
    )

    return out


def replacement_func():
    return meta4d_fused