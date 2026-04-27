"""
Shared Triton kernels for fused Conv2D + GELU (with no-op Dropout eliminated).
All passes must share the SAME replacement_func (unified_conv_gelu_dispatch).
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Depthwise 3x3 Conv + GELU (padding=1, stride=1, dilation=1)
# Weight layout: [C, 1, 3, 3]  ->  contiguous [C, 9]
# Grid: (N*C, ceil(H/TH), ceil(W/TW)) — 2D spatial tiling per (n,c) pair.
# Each program loads 9 separate [TH,TW] sub-tiles and accumulates them.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'TH': 8,  'TW': 8},  num_warps=4),
        triton.Config({'TH': 16, 'TW': 8},  num_warps=8),
        triton.Config({'TH': 8,  'TW': 16}, num_warps=8),
        triton.Config({'TH': 16, 'TW': 16}, num_warps=8),
        triton.Config({'TH': 4,  'TW': 16}, num_warps=4),
        triton.Config({'TH': 16, 'TW': 4},  num_warps=4),
        triton.Config({'TH': 4,  'TW': 8},  num_warps=4),
        triton.Config({'TH': 8,  'TW': 4},  num_warps=4),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def depthwise_conv3x3_gelu_tiled_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W,
    TH: tl.constexpr,
    TW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_th = tl.program_id(1)
    pid_tw = tl.program_id(2)

    n  = pid_nc // C
    c  = pid_nc % C
    h0 = pid_th * TH
    w0 = pid_tw * TW

    bias = tl.load(b_ptr + c).to(tl.float32)

    base_nc = n * C * H * W + c * H * W
    ri = tl.arange(0, TH)   # [TH]
    rj = tl.arange(0, TW)   # [TW]

    acc = tl.zeros([TH, TW], dtype=tl.float32) + bias

    # Load each of the 9 (TH x TW) input sub-tiles and accumulate
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            wt = tl.load(w_ptr + c * 9 + kh * 3 + kw).to(tl.float32)
            ih = h0 + kh - 1 + ri             # [TH] absolute input row
            iw = w0 + kw - 1 + rj             # [TW] absolute input col

            ih2d = tl.expand_dims(ih, 1)      # [TH, 1]
            iw2d = tl.expand_dims(iw, 0)      # [1, TW]

            valid_h = (ih >= 0) & (ih < H)    # [TH]
            valid_w = (iw >= 0) & (iw < W)    # [TW]
            valid2d = tl.expand_dims(valid_h, 1) & tl.expand_dims(valid_w, 0)  # [TH, TW]

            x_ptrs = base_nc + ih2d * W + iw2d
            x_sub  = tl.load(x_ptr + x_ptrs, mask=valid2d, other=0.0).to(tl.float32)

            acc += wt * x_sub

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu_out = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    # Store [TH, TW] output tile
    oh2d  = tl.expand_dims(h0 + ri, 1)   # [TH, 1]
    ow2d  = tl.expand_dims(w0 + rj, 0)   # [1, TW]
    valid_out = (oh2d < H) & (ow2d < W)
    out_ptrs  = base_nc + oh2d * W + ow2d
    tl.store(out_ptr + out_ptrs, gelu_out, mask=valid_out)


# ---------------------------------------------------------------------------
# 1x1 Regular Conv + GELU  (padding=0, stride=1, dilation=1, groups=1)
# Weight layout: [C_out, C_in, 1, 1]  -> contiguous [C_out, C_in]
# Input NCHW: [N, C_in, H, W]   Output NCHW: [N, C_out, H, W]
# Flat kernel: each program handles BLOCK_SIZE consecutive output elements.
# C_in is fixed at 64 for all matching graphs (fastvit).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N', 'C_out', 'H', 'W'],
)
@triton.jit
def conv1x1_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_out, H, W,
    C_IN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C_out * H * W
    mask = offsets < total

    # Decode (batch, co, row, col) from flat output index  [N, C_out, H, W]
    col   = (offsets % W).to(tl.int32)
    row   = (offsets // W % H).to(tl.int32)
    co    = (offsets // (W * H) % C_out).to(tl.int32)
    batch = (offsets // (W * H * C_out)).to(tl.int32)

    # Bias (broadcast within same co group)
    bias = tl.load(b_ptr + co, mask=mask, other=0.0).to(tl.float32)
    acc  = bias

    # Accumulate over C_IN channels (static loop for performance)
    # x[batch, ci, row, col] = x_ptr + batch * C_IN * H * W + ci * H * W + row * W + col
    # w[co,   ci]            = w_ptr + co * C_IN + ci
    for ci in tl.static_range(C_IN):
        x_off  = batch * C_IN * H * W + ci * H * W + row * W + col
        wt_off = co * C_IN + ci
        x_val  = tl.load(x_ptr + x_off,  mask=mask, other=0.0).to(tl.float32)
        wt_val = tl.load(w_ptr + wt_off, mask=mask, other=0.0).to(tl.float32)
        acc    = acc + x_val * wt_val

    # GELU
    gelu_out = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.7071067811865476))

    tl.store(out_ptr + offsets, gelu_out, mask=mask)


# ---------------------------------------------------------------------------
# Unified dispatch — ALL passes must return THIS exact function object
# so the framework's replacement_func_limit never drops any pass.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def unified_conv_gelu_dispatch(bias, weight, x, route):
    """
    Route:
      "dw"   – depthwise 3×3, padding=1, stride=1
      "1x1"  – pointwise (groups=1), padding=0, stride=1, C_in=64
    """
    if route == "dw":
        N, C, H, W = x.shape
        out  = torch.empty_like(x)
        grid = lambda meta: (
            N * C,
            (H + meta['TH'] - 1) // meta['TH'],
            (W + meta['TW'] - 1) // meta['TW'],
        )
        depthwise_conv3x3_gelu_tiled_kernel[grid](
            x, weight, bias, out,
            N, C, H, W,
        )
        return out
    elif route == "1x1":
        N, C_in, H, W = x.shape
        C_out = weight.shape[0]
        out   = torch.empty((N, C_out, H, W), dtype=x.dtype, device=x.device)
        total = N * C_out * H * W
        grid  = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
        conv1x1_gelu_kernel[grid](
            x, weight, bias, out,
            N, C_out, H, W,
            C_IN=C_in,
        )
        return out
    else:
        # Unreachable branch; keeps static analysis happy
        return x