import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'MAX_ITERS': 3}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64, 'MAX_ITERS': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'MAX_ITERS': 2}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128, 'MAX_ITERS': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'MAX_ITERS': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'MAX_ITERS': 1}, num_warps=8),
    ],
    key=['C', 'H', 'W'],
)
@triton.jit
def fused_silu_avgpool_kernel(
    input_ptr,
    output_ptr,
    C,
    H,
    W,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
    MAX_ITERS: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    n = nc_idx // C
    c = nc_idx % C

    base_ptr = input_ptr + n * stride_n + c * stride_c
    hw = H * W

    acc = 0.0

    for iter_id in range(MAX_ITERS):
        start = iter_id * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hw

        h_idx = offsets // W
        w_idx = offsets % W
        ptr_offsets = h_idx * stride_h + w_idx * stride_w

        x = tl.load(base_ptr + ptr_offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        silu_val = x_f32 * tl.sigmoid(x_f32)

        acc += tl.sum(tl.where(mask, silu_val, 0.0), axis=0)

    result = acc / (H * W * 1.0)
    tl.store(output_ptr + nc_idx, result)


@torch.fx.wrap
def fused_silu_avgpool_flatten(in_0):
    N, C, H, W = in_0.shape
    out = torch.empty(N, C, dtype=in_0.dtype, device=in_0.device)

    grid = (N * C,)
    fused_silu_avgpool_kernel[grid](
        in_0, out,
        C, H, W,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
    )
    return out


def replacement_func():
    return fused_silu_avgpool_flatten