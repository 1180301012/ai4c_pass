import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return (tmp_1,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def avg_pool2d_direct_kernel(
    in_ptr,
    out_ptr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    stride_in_b, stride_in_c, stride_in_h, stride_in_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    POOL_H: tl.constexpr,
    POOL_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid // C
    c = pid % C

    h_out_start = pid_h * BLOCK_H
    # 2D offsets for broadcastable indexing
    h_out_offsets = h_out_start + tl.arange(0, BLOCK_H)[:, None]  # (BLOCK_H, 1)
    w_out_offsets = tl.arange(0, BLOCK_W)[None, :]  # (1, BLOCK_W)

    h_mask = h_out_offsets < H_out
    w_mask = w_out_offsets < W_out
    mask = h_mask & w_mask  # (BLOCK_H, BLOCK_W)

    in_base = in_ptr + b * stride_in_b + c * stride_in_c
    out_base = out_ptr + b * stride_out_b + c * stride_out_c

    h_in_offsets = h_out_offsets * POOL_H
    w_in_offsets = w_out_offsets * POOL_W

    sum_val = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for dh in range(POOL_H):
        for dw in range(POOL_W):
            h_idx = h_in_offsets + dh
            w_idx = w_in_offsets + dw
            load_mask = (h_idx < H_in) & (w_idx < W_in) & mask

            ptrs = in_base + h_idx * stride_in_h + w_idx * stride_in_w
            val = tl.load(ptrs, mask=load_mask, other=0.0)
            sum_val += val

    avg_val = sum_val / (POOL_H * POOL_W)

    out_ptrs = out_base + h_out_offsets * stride_out_h + w_out_offsets * stride_out_w
    tl.store(out_ptrs, avg_val, mask=mask)


@triton.jit
def copy_channel_kernel(
    in_ptr,
    out_ptr,
    C_src: tl.constexpr,
    C_dst_offset: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_in_b, stride_in_c, stride_in_h, stride_in_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid // C_src
    c_src = pid % C_src
    c_dst = c_src + C_dst_offset

    h_start = pid_h * BLOCK_H
    # 2D offsets for broadcastable indexing
    h_offsets = h_start + tl.arange(0, BLOCK_H)[:, None]  # (BLOCK_H, 1)
    w_offsets = tl.arange(0, BLOCK_W)[None, :]  # (1, BLOCK_W)

    h_mask = h_offsets < H
    w_mask = w_offsets < W
    mask = h_mask & w_mask  # (BLOCK_H, BLOCK_W)

    in_base = in_ptr + b * stride_in_b + c_src * stride_in_c
    out_base = out_ptr + b * stride_out_b + c_dst * stride_out_c

    ptrs_in = in_base + h_offsets * stride_in_h + w_offsets * stride_in_w
    ptrs_out = out_base + h_offsets * stride_out_h + w_offsets * stride_out_w

    val = tl.load(ptrs_in, mask=mask, other=0.0)
    tl.store(ptrs_out, val, mask=mask)


@torch.fx.wrap
def fused_avg_pool_cat(in_0, in_1):
    B = in_0.shape[0]
    C0 = in_0.shape[1]   # = 20
    C1 = in_1.shape[1]   # = 40
    H_in = in_0.shape[2]  # = 64
    W_in = in_0.shape[3]  # = 48
    H_out = in_1.shape[2] # = 32
    W_out = in_1.shape[3] # = 24
    C_out = C0 + C1       # = 60

    out = torch.empty((B, C_out, H_out, W_out), dtype=in_0.dtype, device=in_0.device)

    BLOCK_H = 8
    BLOCK_W = 32  # must be power of 2, >= W_out=24
    POOL_H = H_in // H_out  # = 2
    POOL_W = W_in // W_out  # = 2

    # Launch avg_pool2d kernel (writes pooled result directly to first C0 channels of output)
    grid_pool = (B * C0, (H_out + BLOCK_H - 1) // BLOCK_H)
    avg_pool2d_direct_kernel[grid_pool](
        in_ptr=in_0,
        out_ptr=out,
        C=C0,
        H_in=H_in, W_in=W_in,
        H_out=H_out, W_out=W_out,
        stride_in_b=in_0.stride(0), stride_in_c=in_0.stride(1),
        stride_in_h=in_0.stride(2), stride_in_w=in_0.stride(3),
        stride_out_b=out.stride(0), stride_out_c=out.stride(1),
        stride_out_h=out.stride(2), stride_out_w=out.stride(3),
        POOL_H=POOL_H, POOL_W=POOL_W,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )

    # Launch copy kernel (copies in_1 to remaining C1 channels of output)
    grid_copy = (B * C1, (H_out + BLOCK_H - 1) // BLOCK_H)
    copy_channel_kernel[grid_copy](
        in_ptr=in_1,
        out_ptr=out,
        C_src=C1, C_dst_offset=C0,
        H=H_out, W=W_out,
        stride_in_b=in_1.stride(0), stride_in_c=in_1.stride(1),
        stride_in_h=in_1.stride(2), stride_in_w=in_1.stride(3),
        stride_out_b=out.stride(0), stride_out_c=out.stride(1),
        stride_out_h=out.stride(2), stride_out_w=out.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
    )

    return out


def replacement_func():
    return fused_avg_pool_cat