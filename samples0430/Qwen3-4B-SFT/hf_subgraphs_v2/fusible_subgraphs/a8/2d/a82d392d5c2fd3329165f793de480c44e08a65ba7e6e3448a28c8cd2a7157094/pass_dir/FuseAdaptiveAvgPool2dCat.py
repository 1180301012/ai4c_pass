import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 16, 'POOL_H': 2, 'POOL_W': 2}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 16, 'POOL_H': 2, 'POOL_W': 2}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 16, 'POOL_H': 2, 'POOL_W': 2}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 8,  'POOL_H': 2, 'POOL_W': 2}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 8,  'POOL_H': 2, 'POOL_W': 2}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'POOL_H': 2, 'POOL_W': 2}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 8,  'POOL_H': 4, 'POOL_W': 2}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'POOL_H': 4, 'POOL_W': 2}, num_stages=2, num_warps=4),
    ],
    key=['B', 'C_pool', 'H_in', 'W_in', 'H_out', 'W_out'],
)
@triton.jit
def avg_pool2d_kernel(
    in0_ptr, out_ptr,
    B, C_pool, H_in, W_in, H_out, W_out,
    POOL_H: tl.constexpr,  # = H_in // H_out, passed as constexpr
    POOL_W: tl.constexpr,  # = W_in // W_out
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Fused adaptive average pool2d kernel.
    Each program handles one (B, C, H_out, W_out) slice.
    Accumulates in float32 for numerical accuracy, then casts back to input dtype.
    POOL_H/POOL_W are constexpr so the inner loops fully unroll at compile time.
    """
    pid = tl.program_id(0)
    C = C_pool
    num_pool_elems = POOL_H * POOL_W

    # Recover (b, c, oh, ow) from flat pid
    oh = pid % H_out
    tmp = pid // H_out
    ow = tmp % W_out
    tmp2 = tmp // W_out
    c = tmp2 % C
    b = tmp2 // C

    base = b * (C * H_in * W_in) + c * (H_in * W_in)

    th_start = oh * POOL_H
    tw_start = ow * POOL_W
    pool_sum = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    # Compute whether each tile element is in-bounds (for generality)
    th = th_start + tl.arange(0, BLOCK_H)
    tw = tw_start + tl.arange(0, BLOCK_W)
    ih = th[:, None] < H_in
    iw = tw[None, :] < W_in
    valid = ih & iw

    # POOL_H and POOL_W are constexpr so Triton unrolls this inner loop
    for th_off in range(POOL_H):
        for tw_off in range(POOL_W):
            x = tl.load(in0_ptr + base
                        + ((th_off + th - th_start[:, None]) * W_in
                          + (tw_off + tw - tw_start[None, :])),
                        mask=valid, other=0.0)
            pool_sum = pool_sum + x.to(tl.float32)

    # Average pooling window elements
    pool_sum = pool_sum / num_pool_elems
    mask_out = (th[:, None] < H_out) & (tw[None, :] < W_out)
    tl.store(out_ptr + b * (C * H_out * W_out) + c * (H_out * W_out)
             + (th[:, None] << 5) + (tw[None, :]), pool_sum.to(out_ptr.dtype.element_ty),
             mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_stages=2, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def copy_kernel(
    in1_ptr, out_ptr, N,
    C_in1, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple elementwise copy kernel filling channels C_pool .. C_pool+C_in1.
    out[b, c, h, w] = in1[b, c-C_pool, h, w]   for c in [C_pool, C_pool+C_in1)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # in1 base stride for channel offset: C_pool channels before -> offset = C_pool * H_out * W_out
    in1_base = C_pool * H_out * W_out
    in1_off = in1_base + (offsets // (H_out * W_out)) * (C_in1 * H_out * W_out) \
             + (offsets % (H_out * W_out)) // W_out * W_out \
             + (offsets % (H_out * W_out)) % W_out

    val = tl.load(in1_ptr + in1_base + in1_off, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def fused_avg_pool_cat(in_0, in_1):
    B, C_pool, H_in, W_in = in_0.shape
    C_in1, H_out, W_out = in_1.shape[1], in_1.shape[2], in_1.shape[3]
    C_total = C_pool + C_in1

    # Output buffer [B, C_total, H_out, W_out]
    out = torch.empty((B, C_total, H_out, W_out), dtype=in_0.dtype, device=in_0.device)

    N_pool = B * C_pool * H_out * W_out

    # --- Kernel 1: fused adaptive_avg_pool2d ---
    avg_pool2d_kernel[lambda meta: (N_pool,)](
        in0_ptr=in_0, out_ptr=out,
        B=B, C_pool=C_pool, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out,
        POOL_H=H_in // H_out, POOL_W=W_in // W_out,
    )

    # --- Kernel 2: copy second half channels ---
    N_copy = B * C_in1 * H_out * W_out
    copy_kernel[lambda meta: ((N_copy + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)](
        in1_ptr=in_1, out_ptr=out,
        N=N_copy, C_in1=C_in1, H_out=H_out, W_out=W_out,
    )

    return out


def replacement_func():
    return fused_avg_pool_cat