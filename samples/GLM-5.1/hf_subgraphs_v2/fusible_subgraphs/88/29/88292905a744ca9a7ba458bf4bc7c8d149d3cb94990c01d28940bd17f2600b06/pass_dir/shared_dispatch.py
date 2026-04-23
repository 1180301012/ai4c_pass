import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['B', 'C', 'H', 'W'],
)
@triton.jit
def fused_softmax_mul_reduce_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, C, H, W,
    stride_in0_b, stride_in0_k, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_k, stride_in1_c,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_spatial = tl.program_id(1)

    b = pid_bc // C
    c = pid_bc % C

    spatial_start = pid_spatial * BLOCK_SIZE
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < H * W

    h = spatial_offsets // W
    w = spatial_offsets % W

    # Load softmax input values (2 values per (b, c) pair)
    # in_1 shape: [B, K=2, 1, C], we access in_1[b, k, 0, c]
    x0 = tl.load(in_1_ptr + b * stride_in1_b + c * stride_in1_c).to(tl.float32)
    x1 = tl.load(in_1_ptr + b * stride_in1_b + stride_in1_k + c * stride_in1_c).to(tl.float32)

    # Numerically stable softmax in float32
    max_x = tl.maximum(x0, x1)
    e0 = tl.exp(x0 - max_x)
    e1 = tl.exp(x1 - max_x)
    sum_e = e0 + e1
    w0 = e0 / sum_e
    w1 = e1 / sum_e

    # Load feature values for both classes
    # in_0 shape: [B, K=2, C, H, W], we access in_0[b, k, c, h, w]
    base_offset = b * stride_in0_b + c * stride_in0_c + h * stride_in0_h + w * stride_in0_w
    v0 = tl.load(in_0_ptr + base_offset, mask=mask).to(tl.float32)
    v1 = tl.load(in_0_ptr + base_offset + stride_in0_k, mask=mask).to(tl.float32)

    # Weighted sum
    result = w0 * v0 + w1 * v1

    # Store (auto-converts to output dtype)
    tl.store(out_ptr + b * stride_out_b + c * stride_out_c + h * stride_out_h + w * stride_out_w, result, mask=mask)


@torch.fx.wrap
def fused_softmax_mul_reduce(in_0, in_1):
    B, K, C, H, W = in_0.shape
    assert K == 2

    out = torch.empty((B, C, H, W), dtype=in_0.dtype, device=in_0.device)

    HW = H * W
    grid = lambda meta: (B * C, triton.cdiv(HW, meta['BLOCK_SIZE']))

    fused_softmax_mul_reduce_kernel[grid](
        in_0, in_1, out,
        B, C, H, W,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3), in_0.stride(4),
        in_1.stride(0), in_1.stride(1), in_1.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )

    return out


@torch.fx.wrap
def dispatch_wrapper(in_0, in_1, route):
    # All routes call the same kernel since it handles arbitrary shapes
    if route == "route_8":
        return fused_softmax_mul_reduce(in_0, in_1)
    elif route == "route_1":
        return fused_softmax_mul_reduce(in_0, in_1)
    elif route == "route_2":
        return fused_softmax_mul_reduce(in_0, in_1)
    else:
        raise ValueError(f"Unknown route: {route}")