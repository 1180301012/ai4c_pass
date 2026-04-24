"""
Shared Triton kernels for fused depthwise convolution + GELU.
All conv+gelu+dropout pass files import from here.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def _dw_conv3x3_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused depthwise 3x3 conv + GELU.
    Grid: (N*C, ceil(H*W / BLOCK_SIZE))
    """
    nc = tl.program_id(0)
    sp = tl.program_id(1)

    n = nc // C
    c = nc % C

    sp_start = sp * BLOCK_SIZE
    hw_offs = sp_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offs < H * W

    oh = hw_offs // W
    ow = hw_offs % W

    bias = tl.load(b_ptr + c).to(tl.float32)
    x_base = x_ptr + n * (C * H * W) + c * (H * W)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for kh in range(3):
        for kw in range(3):
            ih = oh + kh - 1
            iw = ow + kw - 1
            in_valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
            ih_s = tl.maximum(tl.minimum(ih, H - 1), 0)
            iw_s = tl.maximum(tl.minimum(iw, W - 1), 0)
            x_val = tl.load(x_base + ih_s * W + iw_s, mask=in_valid, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptr + c * 9 + kh * 3 + kw).to(tl.float32)
            acc += x_val * w_val

    acc += bias
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    out_base = out_ptr + n * (C * H * W) + c * (H * W)
    tl.store(out_base + oh * W + ow, gelu.to(x_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_NC': 16, 'BLOCK_SP': 64}, num_warps=4),
        triton.Config({'BLOCK_NC': 32, 'BLOCK_SP': 64}, num_warps=4),
        triton.Config({'BLOCK_NC': 64, 'BLOCK_SP': 64}, num_warps=8),
        triton.Config({'BLOCK_NC': 16, 'BLOCK_SP': 128}, num_warps=4),
        triton.Config({'BLOCK_NC': 32, 'BLOCK_SP': 128}, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def _dw_conv1x1_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_out, C_in, H, W,
    BLOCK_NC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    """Fused 1x1 depthwise conv + GELU (general C_in).
    Grid: (ceil(N*C_out / BLOCK_NC), ceil(H*W / BLOCK_SP))
    """
    nc_start = tl.program_id(0) * BLOCK_NC
    sp_start = tl.program_id(1) * BLOCK_SP

    nc_offs = nc_start + tl.arange(0, BLOCK_NC)   # [BLOCK_NC]
    sp_offs = sp_start + tl.arange(0, BLOCK_SP)   # [BLOCK_SP]

    n_v   = nc_offs // C_out
    c_out_v = nc_offs % C_out

    sp_mask = sp_offs < H * W
    oh = sp_offs // W
    ow = sp_offs % W

    # Pre-load bias [BLOCK_NC]
    b_vals = tl.load(b_ptr + c_out_v, mask=nc_offs < N * C_out, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_NC, BLOCK_SP], dtype=tl.float32)

    for cin in range(C_in):
        # x[n_v[i], cin, oh[j], ow[j]]: shape [BLOCK_NC, BLOCK_SP]
        x_idx = (n_v[:, None] * (C_in * H * W)
                 + cin * (H * W)
                 + oh[None, :] * W
                 + ow[None, :])
        x_mask = (nc_offs[:, None] < N * C_out) & sp_mask[None, :]
        x_vals = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0).to(tl.float32)

        # w[c_out_v[i], cin]: shape [BLOCK_NC]
        w_vals = tl.load(w_ptr + c_out_v * C_in + cin,
                         mask=nc_offs < N * C_out, other=0.0).to(tl.float32)

        acc += x_vals * w_vals[:, None]

    acc += b_vals[:, None]
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    out_idx = (n_v[:, None] * (C_out * H * W)
               + c_out_v[:, None] * (H * W)
               + oh[None, :] * W
               + ow[None, :])
    out_mask = (nc_offs[:, None] < N * C_out) & sp_mask[None, :]
    tl.store(out_ptr + out_idx, gelu.to(x_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def fused_conv_gelu_dispatch(in_0, in_1, in_2, route):
    """
    Dispatch wrapper for fused conv+gelu kernels.
    in_0: bias   [C]
    in_1: weight [C, ...]
    in_2: input  [N, C, H, W]
    route: string selecting the kernel variant
    """
    if route == "groups128_pad1":
        N, C, H, W = in_2.shape
        out = torch.empty_like(in_2)
        grid = lambda meta: (N * C, triton.cdiv(H * W, meta['BLOCK_SIZE']))
        _dw_conv3x3_gelu_kernel[grid](in_2, in_1, in_0, out, N, C, H, W)
        return out
    elif route == "groups256_pad1":
        N, C, H, W = in_2.shape
        out = torch.empty_like(in_2)
        grid = lambda meta: (N * C, triton.cdiv(H * W, meta['BLOCK_SIZE']))
        _dw_conv3x3_gelu_kernel[grid](in_2, in_1, in_0, out, N, C, H, W)
        return out
    elif route == "groups512_pad1":
        N, C, H, W = in_2.shape
        out = torch.empty_like(in_2)
        grid = lambda meta: (N * C, triton.cdiv(H * W, meta['BLOCK_SIZE']))
        _dw_conv3x3_gelu_kernel[grid](in_2, in_1, in_0, out, N, C, H, W)
        return out
    elif route == "groups2048_pad1":
        N, C, H, W = in_2.shape
        out = torch.empty_like(in_2)
        grid = lambda meta: (N * C, triton.cdiv(H * W, meta['BLOCK_SIZE']))
        _dw_conv3x3_gelu_kernel[grid](in_2, in_1, in_0, out, N, C, H, W)
        return out
    elif route == "groups1_pad0":
        # fastvit: weight [256, 64, 1, 1], input [N, 64, H, W]
        N, C_in, H, W = in_2.shape
        C_out = in_1.shape[0]
        out = torch.empty_like(in_2)
        grid = lambda meta: (triton.cdiv(N * C_out, meta['BLOCK_NC']),
                             triton.cdiv(H * W, meta['BLOCK_SP']))
        _dw_conv1x1_gelu_kernel[grid](in_2, in_1, in_0, out, N, C_out, C_in, H, W)
        return out
    elif route == "groups1_pad1":
        # 1x1 depthwise: weight [C, C, 1, 1], input [N, C, H, W]
        N, C, H, W = in_2.shape
        out = torch.empty_like(in_2)
        grid = lambda meta: (triton.cdiv(N * C, meta['BLOCK_NC']),
                             triton.cdiv(H * W, meta['BLOCK_SP']))
        _dw_conv1x1_gelu_kernel[grid](in_2, in_1, in_0, out, N, C, C, H, W)
        return out
    else:
        out = torch.empty_like(in_2)
        return out