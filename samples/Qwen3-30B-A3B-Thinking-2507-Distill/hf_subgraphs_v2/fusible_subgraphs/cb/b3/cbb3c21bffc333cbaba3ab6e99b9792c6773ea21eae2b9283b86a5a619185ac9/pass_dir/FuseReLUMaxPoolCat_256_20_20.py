import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
    ],
    key=['total_HW'],
)
@triton.jit
def fused_relu_maxpool_cat_kernel(
    in_ptr,
    rp_ptr,
    out_ptr,
    N, C, H_out, W_out, total_HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused ReLU + MaxPool2d(kernel=5,stride=1,pad=2) + Cat kernel.

    Grid: (2 * N * C,)
    pid_nc <  N*C  -> handles relu/pool output (writes to channels 0..C-1)
    pid_nc >= N*C  -> handles copy_pool output (writes to channels C..4C-1)

    For copy_pool: pool_part = (pid_nc - N*C) // (N*C)
                   c_in       = (pid_nc - N*C) % (N*C)
    """
    pid_nc = tl.program_id(0)
    N_tC   = N * C

    if pid_nc < N_tC:
        n       = pid_nc // C
        c       = pid_nc % C
        in_base = (n * C + c) * total_HW
        out_base = (n * 4 * C + c) * total_HW
    else:
        pool_part = (pid_nc - N_tC) // N_tC
        nc_off    = (pid_nc - N_tC) % N_tC
        n         = nc_off // C
        c         = nc_off % C
        in_base   = (n * C + c) * total_HW
        out_base  = (n * 4 * C + C + pool_part * C + c) * total_HW

    hw_off  = tl.arange(0, BLOCK_HW)
    hw_mask = hw_off < total_HW

    h_out = hw_off // W_out
    w_out = hw_off % W_out

    # Load input pixel values; handle out-of-bounds with mask
    in_idx = in_base + hw_off
    x = tl.load(in_ptr + in_idx, mask=hw_mask, other=-1e9)

    # ReLU
    x_relu = tl.maximum(x, 0.0)

    # Write relu/pool result (only for pid_nc < N_tC branch)
    if pid_nc < N_tC:
        tl.store(rp_ptr + in_base + hw_off, x_relu, mask=hw_mask)
    else:
        # Copy to third, fourth quarter — these are redundant copies of the maxpool
        tl.store(out_ptr + out_base + hw_off, x_relu, mask=hw_mask)
        return

    # Compute max_pool2d(kernel=5, stride=1, padding=2)
    pool_max = tl.full([BLOCK_HW], -1e9, dtype=tl.float32)

    for kh in range(5):
        for kw in range(5):
            h_in = h_out + kh - 2
            w_in = w_out + kw - 2
            in_range = (h_in >= 0) & (h_in < H_out) & (w_in >= 0) & (w_in < W_out)
            valid = hw_mask & in_range
            in_idx = in_base + h_in * W_out + w_in
            x_val = tl.load(in_ptr + in_idx, mask=valid, other=-1e9)
            x_val = tl.maximum(x_val, 0.0)   # max with relu
            pool_max = tl.maximum(pool_max, x_val)

    # Write maxpool result to second quarter channels
    rp_out_base = (n * 4 * C + C + c) * total_HW
    tl.store(rp_ptr + rp_out_base + hw_off, pool_max, mask=hw_mask)


@torch.fx.wrap
def fused_relu_maxpool_cat(in_0):
    N  = in_0.shape[0]
    C  = in_0.shape[1]
    H  = in_0.shape[2]
    W  = in_0.shape[3]
    # Output: [N, 4*C, H, W]
    out  = torch.empty((N, 4 * C, H, W), dtype=in_0.dtype, device=in_0.device)
    # Intermediate relu+pool output, reused by copy-pool part
    rp   = torch.empty((N, C, H, W),     dtype=in_0.dtype, device=in_0.device)

    total_HW = H * W
    grid = lambda meta: (2 * N * C,)

    fused_relu_maxpool_cat_kernel[grid](
        in_0, rp, out,
        N, C, H, W, total_HW,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface expected by the AI4C framework
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_relu_maxpool_cat