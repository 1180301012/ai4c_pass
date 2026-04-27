import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: fused sigmoid(x[B,C,1,1]) * y[B,C,H,W]  -->  out[B,C,H,W]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def _sigmoid_mul_kernel(
    x_ptr,      # [B, C, 1, 1]  -> stride [C,1,1,1]
    y_ptr,      # [B, C, H, W]  -> contiguous
    out_ptr,    # [B, C, H, W]  -> contiguous
    B, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    N    = B * C * HW
    mask = offs < N

    hw    = offs % HW
    c     = (offs // HW) % C
    b_idx = offs // (C * HW)

    # x[b,c,0,0] lives at flat offset b*C + c  (since last two dims are 1)
    x_val = tl.load(x_ptr + b_idx * C + c, mask=mask, other=0.0)
    y_val = tl.load(y_ptr + offs,           mask=mask, other=0.0)

    sig_x   = tl.sigmoid(x_val.to(tl.float32)).to(y_val.dtype)
    out_val = y_val * sig_x
    tl.store(out_ptr + offs, out_val, mask=mask)


@torch.fx.wrap
def sigmoid_mul_triton(x, y):
    """
    x : [B, C, 1, 1]  (conv2d output, will be sigmoid'd)
    y : [B, C, H, W]
    returns sigmoid(x) * y  of shape [B, C, H, W]
    """
    B, C   = x.shape[0], x.shape[1]
    HW     = y.shape[2] * y.shape[3]
    N      = B * C * HW
    out    = torch.empty_like(y)
    grid   = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _sigmoid_mul_kernel[grid](x, y, out, B, C, HW)
    return out


# ---------------------------------------------------------------------------
# Kernel 2: channel-shuffle-chunk
#   a, b : [B, C, H, W]  (two equally-shaped tensors)
#   Implements: cat([a,b],dim=1).view(B,2,C,H,W).transpose(1,2).contiguous()
#               .view(B,2C,H,W).chunk(2,dim=1)
#
#   out0[b, k, hw] = a[b, k//2, hw]       if k%2==0
#                  = b[b, k//2, hw]       if k%2==1      k in [0, C)
#   out1[b, k, hw] = a[b, C//2+k//2, hw] if k%2==0
#                  = b[b, C//2+k//2, hw] if k%2==1      k in [0, C)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def _channel_shuffle_chunk_kernel(
    a_ptr, b_ptr,
    out0_ptr, out1_ptr,
    B, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    N    = B * C * HW
    mask = offs < N

    hw    = offs % HW
    k     = (offs // HW) % C       # output channel in [0, C)
    b_idx = offs // (C * HW)

    is_a    = ((k & 1) == 0)       # even output channels come from a
    half_C  = C // 2
    src_c0  = k >> 1               # [0, C/2)
    src_c1  = half_C + (k >> 1)   # [C/2, C)

    src_off0 = b_idx * (C * HW) + src_c0 * HW + hw
    src_off1 = b_idx * (C * HW) + src_c1 * HW + hw

    # --- out0 ---
    va0  = tl.load(a_ptr + src_off0, mask=mask &  is_a, other=0.0)
    vb0  = tl.load(b_ptr + src_off0, mask=mask & ~is_a, other=0.0)
    tl.store(out0_ptr + offs, tl.where(is_a, va0, vb0), mask=mask)

    # --- out1 ---
    va1  = tl.load(a_ptr + src_off1, mask=mask &  is_a, other=0.0)
    vb1  = tl.load(b_ptr + src_off1, mask=mask & ~is_a, other=0.0)
    tl.store(out1_ptr + offs, tl.where(is_a, va1, vb1), mask=mask)


@torch.fx.wrap
def channel_shuffle_chunk_triton(a, b):
    """
    a, b : [B, C, H, W]  (same shape, channels interleaved)
    returns (out0, out1) each of shape [B, C, H, W]
    """
    B, C, H, W = a.shape
    HW   = H * W
    N    = B * C * HW
    out0 = torch.empty_like(a)
    out1 = torch.empty_like(a)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _channel_shuffle_chunk_kernel[grid](a, b, out0, out1, B, C, HW)
    return out0, out1


# ---------------------------------------------------------------------------
# Kernel 3: simple channel interleave (single output)
#   a, b : [B, C, H, W]  →  out [B, 2C, H, W]  where
#   out[b, 2i, h, w]   = a[b, i, h, w]
#   out[b, 2i+1, h, w] = b[b, i, h, w]
# Single output avoids multi-output FX replacement issues.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def _channel_shuffle_kernel(
    a_ptr, b_ptr, out_ptr,
    B, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    N    = B * 2 * C * HW
    mask = offs < N

    hw    = offs % HW
    out_c = (offs // HW) % (2 * C)
    b_idx = offs // (2 * C * HW)

    from_a  = ((out_c & 1) == 0)
    src_c   = out_c >> 1
    src_off = b_idx * C * HW + src_c * HW + hw

    va = tl.load(a_ptr + src_off, mask=mask &  from_a, other=0.0)
    vb = tl.load(b_ptr + src_off, mask=mask & ~from_a, other=0.0)
    tl.store(out_ptr + offs, tl.where(from_a, va, vb), mask=mask)


@torch.fx.wrap
def channel_shuffle_triton(a, b):
    """
    a, b : [B, C, H, W]
    returns : [B, 2C, H, W] with channels interleaved (single tensor output)
    """
    B, C, H, W = a.shape
    HW   = H * W
    N    = B * 2 * C * HW
    out  = torch.empty(B, 2 * C, H, W, dtype=a.dtype, device=a.device)
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _channel_shuffle_kernel[grid](a, b, out, B, C, HW)
    return out


# ---------------------------------------------------------------------------
# Unified dispatch wrapper (routing technique)
# ALL pass files return this single function so replacement_func_limit == 1
#   route="sigmoid_mul"  -> sigmoid(a) * b  (single tensor)
#   route="shuffle"      -> channel_shuffle_triton(a,b)  (single tensor)
# NOT @torch.fx.wrap so FX traces into it. Both branches return single tensors
# so FX can map one output node to one pattern output without getitem issues.
# ---------------------------------------------------------------------------
def dispatch(a, b, route):
    if route == "sigmoid_mul":
        return sigmoid_mul_triton(a, b)
    else:
        return channel_shuffle_triton(a, b)