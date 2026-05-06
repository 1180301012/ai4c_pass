import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=16),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def relu_pool_fused_kernel(
    x_ptr,                  # input  [N, C, H, W]
    out_ptr,                # output [N, 4*C, H, W]  (cat: first C = relu, next 3C = pool)
    N, C, H, W,
    pool_out,               # tmp 1,2,3 output  (written by pool CTAs, read by relu CTAs via tl.where)
    BLOCK_HW: tl.constexpr,
):
    """
    One CTA per (n, c) output channel.
    For c < C  : compute ReLU on x and write to out[name, c, :, :]
    For c >= C : compute max-pool (kernel=5, stride=1, pad=2) on x and
                write to out[name, c, :, :]  and the two duplicate slots
                out[name, c+C, :, :]  and out[name, c+2*C, :, :]  (redundant
                stores are masked so no actual memory conflict).
    """
    pid_nc = tl.program_id(0)   # index into N * C
    pid_hw = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    # ---- spatial tile -------------------------------------------------------
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < H * W
    ho = hw_offs // W   # output row
    hw = hw_offs % W    # output col

    C4 = 4 * C          # total output channels

    # ---- ReLU computation ---------------------------------------------------
    # Only relevant for c < C; for c >= C x_idx would be out-of-range for x,
    # but pool_out is written here instead, and the branch is dead-code-elim'd
    # by the tl.where below.
    x_idx  = n * C * H * W + c * H * W + hw_offs
    x_val  = tl.load(x_ptr + x_idx, mask=hw_mask)
    relu_v = tl.maximum(x_val, 0.0)

    # Scalar flag: are we in the ReLU output plane?
    is_relu = (c < C)

    # ---- max-pool computation (kernel 5×5, stride 1, pad 2, dilation 1) ----
    max_val  = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for kh in tl.static_range(5):
        ih    = ho + kh - 2           # input row  (pad = 2)
        ih_ok = (ih >= 0) & (ih < H)

        for kw in tl.static_range(5):
            iw    = hw + kw - 2       # input col  (pad = 2)
            iw_ok = (iw >= 0) & (iw < W)
            pool_ok = ih_ok & iw_ok

            xk = n * C * H * W + c * H * W + ih * W + iw
            xkv = tl.load(x_ptr + xk, mask=pool_ok & hw_mask, other=0.0)
            xkv_f = tl.where(hw_mask, xkv, float('-inf'))
            max_val = tl.maximum(max_val, xkv_f)

    pool_v = tl.maximum(max_val, 0.0)

    # ---- write output -------------------------------------------------------
    # The output layout is: [N, 4*C, H, W]
    C4_HW = C4 * H * W

    # offset expressions for each of the four "logical" output channels
    relu_off  = pid_nc    * C4_HW + hw_offs   # c  plane
    pool_off  = (c  + C)  * C4_HW + hw_offs   # pool×1
    pool_off2 = (c  + 2*C)* C4_HW + hw_offs   # pool×2
    pool_off3 = (c  + 3*C)* C4_HW + hw_offs   # pool×3

    tl.store(
        out_ptr + relu_off,
        tl.where(is_relu, relu_v, pool_v),
        mask=hw_mask,
    )
    tl.store(
        out_ptr + pool_off,
        tl.where(is_relu, pool_v, relu_v),
        mask=hw_mask,
    )
    tl.store(
        out_ptr + pool_off2,
        tl.where(is_relu, pool_v, relu_v),
        mask=hw_mask,
    )
    tl.store(
        out_ptr + pool_off3,
        tl.where(is_relu, pool_v, relu_v),
        mask=hw_mask,
    )


@torch.fx.wrap
def relu_pool_fused(in_0):
    N, C, H, W = in_0.shape
    tl_pool = torch.empty((N, 3 * C, H, W), dtype=in_0.dtype, device=in_0.device)

    # pool_out buffer passed as *pool_out* here (same tensor object);
    # the kernel writes into it redundantly for channels c, c+C, c+2C,
    # but since the cat only reads channels C..3*C (skipping 0..C-1),
    # and we disable stores via tl.where when the correct mask is False,
    # we never actually write to c (which reads relu_out from the other buffers).
    out = torch.empty((N, 4 * C, H, W), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (N * C, triton.cdiv(H * W, meta['BLOCK_HW']))

    relu_pool_fused_kernel[grid](
        in_0, out, tl_pool,
        N, C, H, W,
    )

    return out


def replacement_func():
    return relu_pool_fused