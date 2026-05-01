import torch
import triton
import triton.language as tl


def pattern(conv_out, in_2):
    tmp_3 = conv_out.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(conv_out, in_2):
    return (conv_out, in_2)


# ─── 2-D kernel: no integer division; one block covers BLOCK_SIZE spatial positions
#      for a single (n,c) pair.  Best when NC is large (N=32).
@triton.jit
def sigmoid_mul_hardtanh_2d_kernel(
    conv_out_ptr,
    in2_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    nc_pid  = tl.program_id(0)   # (n,c) index
    hw_pid  = tl.program_id(1)   # chunk index along H*W

    # One scalar sigmoid per (n,c) – no gather, no division
    scale = tl.sigmoid(tl.load(conv_out_ptr + nc_pid).to(tl.float32))

    base      = nc_pid * HW
    hw_start  = tl.multiple_of(hw_pid * BLOCK_SIZE, BLOCK_SIZE)
    offsets   = tl.max_contiguous(hw_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    mask      = offsets < HW

    vals   = tl.load(in2_ptr + base + offsets, mask=mask, other=0.0)
    result = tl.minimum(tl.maximum(vals.to(tl.float32) * scale, 0.0), 6.0)
    tl.store(out_ptr + base + offsets, result.to(vals.dtype), mask=mask)


# ─── Flat 1-D kernel: HW is constexpr → compiler turns `// HW` into a
#      multiply-shift sequence (no actual division instruction).
@triton.jit
def sigmoid_mul_hardtanh_flat_kernel(
    conv_out_ptr,
    in2_ptr,
    out_ptr,
    total,
    HW:         tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid     = tl.program_id(0)
    base    = tl.multiple_of(pid * BLOCK_SIZE, BLOCK_SIZE)
    offsets = tl.max_contiguous(base + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    mask    = offsets < total

    vals       = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    nc_idx     = offsets // HW          # efficient: HW is constexpr
    scale_raw  = tl.load(conv_out_ptr + nc_idx, mask=mask, other=0.0)
    scale_f32  = tl.sigmoid(scale_raw.to(tl.float32))

    result = tl.minimum(tl.maximum(vals.to(tl.float32) * scale_f32, 0.0), 6.0)
    tl.store(out_ptr + offsets, result.to(vals.dtype), mask=mask)


@torch.fx.wrap
def triton_sigmoid_mul_hardtanh(conv_out, in_2):
    # conv_out: [N, C, 1, 1],  in_2: [N, C, H, W]
    N  = conv_out.shape[0]
    C  = conv_out.shape[1]
    H  = in_2.shape[2]
    W  = in_2.shape[3]
    HW = H * W
    NC = N * C
    total = NC * HW

    out = torch.empty_like(in_2)

    # ── Large-batch path (NC large enough to fill the GPU via the first grid dim)
    # Use the 2-D kernel: avoids integer division, uses scalar sigmoid per block.
    # BLOCK_SIZE=1024 → 8 float16/thread (128-bit load) with num_warps=4.
    if NC >= 512:
        if HW <= 1024:
            # 2-D kernel: one block per (n,c), BLOCK_SIZE ≥ HW (at most 24 % mask waste)
            BLOCK_SIZE_2D = 1024
            grid_2d = (NC, triton.cdiv(HW, BLOCK_SIZE_2D))
            sigmoid_mul_hardtanh_2d_kernel[grid_2d](
                conv_out, in_2, out, HW, BLOCK_SIZE_2D,
                num_warps=4,
            )
        else:
            # Large HW (e.g. 2304): flat 1-D is more bandwidth-efficient
            # BLOCK_SIZE=4096 → ~4 waves, good pipelining, 99 % fill
            BLOCK_SIZE_FLAT = 4096
            grid_flat = (triton.cdiv(total, BLOCK_SIZE_FLAT),)
            sigmoid_mul_hardtanh_flat_kernel[grid_flat](
                conv_out, in_2, out,
                total, HW, BLOCK_SIZE_FLAT,
                num_warps=8,
            )

    # ── Small-batch path (NC=228 for N=1): use flat 1-D for more parallelism
    else:
        if total <= 1024 * 1024:      # ≤ 1 M elements
            BLOCK_SIZE_SB = 512
            nw_sb         = 4
        else:
            BLOCK_SIZE_SB = 1024
            nw_sb         = 4
        grid_sb = (triton.cdiv(total, BLOCK_SIZE_SB),)
        sigmoid_mul_hardtanh_flat_kernel[grid_sb](
            conv_out, in_2, out,
            total, HW, BLOCK_SIZE_SB,
            num_warps=nw_sb,
        )

    return out


def replacement_func():
    return triton_sigmoid_mul_hardtanh