import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['C_in', 'H', 'W'],
)
@triton.jit
def fuse_cat_slice_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, C_in, H, W, C_slice,
    stride_in0_b, stride_in0_c, stride_in0_h, stride_in0_w,
    stride_in1_b, stride_in1_c, stride_in1_h, stride_in1_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = pid // C_slice
    pid_c = pid % C_slice

    # Select source input and channel
    if pid_c < C_in:
        src_ptr = in0_ptr
        src_c = pid_c
        sb = stride_in0_b
        sc = stride_in0_c
        sh = stride_in0_h
        sw = stride_in0_w
    else:
        src_ptr = in1_ptr
        src_c = pid_c - C_in
        sb = stride_in1_b
        sc = stride_in1_c
        sh = stride_in1_h
        sw = stride_in1_w

    src_base = pid_b * sb + src_c * sc
    out_base = pid_b * stride_out_b + pid_c * stride_out_c

    n_tiles = tl.cdiv(HW, BLOCK_SIZE)

    for tile_id in range(n_tiles):
        hw_offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = hw_offsets < HW

        h_idx = hw_offsets // W
        w_idx = hw_offsets % W

        src_off = src_base + h_idx * sh + w_idx * sw
        val = tl.load(src_ptr + src_off, mask=mask, other=0.0)

        out_off = out_base + h_idx * stride_out_h + w_idx * stride_out_w
        tl.store(out_ptr + out_off, val, mask=mask)


@torch.fx.wrap
def fuse_cat_slice_dispatch(in_0, in_1, route):
    B, C_in, H, W = in_0.shape

    if route == "route_120":
        C_slice = 120
    elif route == "route_480":
        C_slice = 480
    elif route == "route_672":
        C_slice = 672
    elif route == "route_960":
        C_slice = 960
    else:
        raise ValueError(f"Unknown route: {route}")

    HW = H * W

    out = torch.empty(B, C_slice, H, W, dtype=in_0.dtype, device=in_0.device)

    grid = (B * C_slice,)

    fuse_cat_slice_kernel[grid](
        in_0, in_1, out,
        B, C_in, H, W, C_slice,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        HW=HW,
    )

    return out


def replacement_func():
    return fuse_cat_slice_dispatch