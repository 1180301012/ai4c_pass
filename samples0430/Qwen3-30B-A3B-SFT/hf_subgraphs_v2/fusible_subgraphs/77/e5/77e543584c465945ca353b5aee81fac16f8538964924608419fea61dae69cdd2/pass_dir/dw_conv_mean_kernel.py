"""
Shared Triton kernels + dispatch wrapper for fused depthwise Conv2D + spatial mean.

Depthwise conv: out[n,c,h,w] = sum_{kh,kw} w[c,0,kh,kw] * x[n,c,h+kh-1,w+kw-1]
Spatial mean:  mean[n,c]    = mean(out[n,c,:,:])

Grid layout for the depthwise conv kernel:
  pid_0 = n * C + c  (each program handles one (batch, channel) pair)
  pid_1 = hw_tile    (tile index within H_out * W_out)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['C', 'HW_out'],
)
@triton.jit
def dw_conv_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, H, W,
    H_out, W_out, HW_out,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused depthwise 3x3 conv (pad=1, kernel=3x3). Grid = (N*C, tiles)."""
    nc_id   = tl.program_id(0)
    hw_tile = tl.program_id(1)

    n_idx = nc_id // C
    c_idx = nc_id % C

    hw_start   = hw_tile * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offsets < HW_out

    h_out = hw_offsets // W_out
    w_out = hw_offsets % W_out

    x_base = n_idx * C * H * W + c_idx * H * W

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Unrolled 3x3 depthwise convolution (static loop unrolling)
    for kh in range(3):
        for kw in range(3):
            h_in = h_out * stride_h - pad_h + kh
            w_in = w_out * stride_w - pad_w + kw

            in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            valid = in_bounds & mask

            x_idx = x_base + h_in * W + w_in
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)

            # weight shape: (C, 1, 3, 3) → offset = c*9 + kh*3 + kw
            w_val = tl.load(w_ptr + c_idx * 9 + kh * 3 + kw).to(tl.float32)

            acc = acc + x_val * w_val

    out_base = n_idx * C * H_out * W_out + c_idx * H_out * W_out
    tl.store(out_ptr + out_base + hw_offsets, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
    ],
    key=['HW_out'],
)
@triton.jit
def dw_mean_reduce_kernel(
    conv_ptr, mean_ptr, count_ptr,
    HW_out,
    BLOCK_SIZE: tl.constexpr,
):
    """Atomic reduce: sum conv values for each (n,c) pair to compute spatial mean."""
    nc_id   = tl.program_id(0)
    hw_tile = tl.program_id(1)

    hw_start = hw_tile * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offsets < HW_out

    ptr = nc_id * HW_out + hw_offsets
    vals = tl.load(conv_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
    chunk_sum = tl.sum(vals, axis=0)

    tl.atomic_add(mean_ptr + nc_id, chunk_sum)
    # Guard: only one tile writes the count marker (hw_tile == 0)
    if hw_tile == 0:
        tl.store(count_ptr + nc_id, tl.cast(HW_out, tl.float32))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW_out'],
)
@triton.jit
def dw_mean_scale_kernel(
    mean_ptr, count_ptr,
    N_C,
    BLOCK_SIZE: tl.constexpr,
):
    """Divide accumulated sum by count to get the mean for each (n,c) pair."""
    pid      = tl.program_id(0)
    offsets  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < N_C

    s = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(count_ptr + offsets, mask=mask, other=1.0)

    tl.store(mean_ptr + offsets, s / c, mask=mask)



@torch.fx.wrap
def _triton_dw_conv(x, w, H, W, H_out, W_out):
    """Depthwise conv → single output tensor [N, C, H_out, W_out]."""
    N, C, _, _ = x.shape
    w_dev  = w.to(x.device)
    NC     = N * C
    out    = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    grid1  = (H_out * W_out + 511) // 512
    dw_conv_kernel[(NC, grid1)](
        x, w_dev, out,
        N, C, H, W, H_out, W_out, H_out * W_out,
        1, 1, 1, 1,
    )
    return out


@torch.fx.wrap
def _triton_mean_reduce_scale(conv_out, N, C, H_out, W_out):
    """Atomic-reduce conv output → spatial mean (N, C, 1, 1)."""
    NC       = N * C
    HW_out   = H_out * W_out
    mean_f32  = torch.zeros((NC,),               dtype=torch.float32, device=conv_out.device)
    count_f32 = torch.ones((NC,),                dtype=torch.float32, device=conv_out.device)
    REDUCE_BLOCK = 256
    reduce_tiles = (HW_out + REDUCE_BLOCK - 1) // REDUCE_BLOCK
    dw_mean_reduce_kernel[(NC, reduce_tiles)](
        conv_out, mean_f32, count_f32, HW_out,
    )
    scale_BLOCK = 256
    scale_tiles = (NC + scale_BLOCK - 1) // scale_BLOCK
    dw_mean_scale_kernel[(scale_tiles,)](
        mean_f32, count_f32, NC,
    )
    return mean_f32.to(conv_out.dtype).view(N, C, 1, 1)


# dispatch_kernel is @torch.fx.wrap'd so FX creates ONE call_function node for it.
# It returns a single tensor (either conv output or mean output) based on route.
@torch.fx.wrap
def dispatch_kernel(x, w, route):
    N, C, H, W = x.shape
    # Heuristic to detect stride from input/output size (pad=1, K=3)
    H_out_std = H
    W_out_std = W
    H_out_str2 = (H + 2 - 3) // 2 + 1
    W_out_str2 = (W + 2 - 3) // 2 + 1
    if (H * W >= 2 * H_out_std * W_out_std
            and H_out_str2 > 0 and W_out_str2 > 0):
        H_out, W_out = H_out_str2, W_out_str2
    else:
        H_out, W_out = H_out_std, W_out_std

    if route == "conv":
        return _triton_dw_conv(x, w, H, W, H_out, W_out)
    else:  # route == "mean"
        conv_out = _triton_dw_conv(x, w, H, W, H_out, W_out)
        return _triton_mean_reduce_scale(conv_out, N, C, H_out, W_out)


@torch.fx.wrap
def dispatch_mean(x):
    """Spatial mean: x.mean((2,3), keepdim=True). Single-output, @torch.fx.wrap'd."""
    N, C, H, W = x.shape
    H_out, W_out = H, W
    return _triton_mean_reduce_scale(x, N, C, H_out, W_out)