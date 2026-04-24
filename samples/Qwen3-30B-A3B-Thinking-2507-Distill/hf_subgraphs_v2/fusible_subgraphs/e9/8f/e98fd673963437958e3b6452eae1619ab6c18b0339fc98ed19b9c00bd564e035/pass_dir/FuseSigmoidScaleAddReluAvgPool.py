import torch
import triton
import triton.language as tl


# Pass 1: fuse sigmoid -> view -> expand_as -> mul
# Produces scaled [B, C, H, W] tensor; iadd runs natively after.
def pattern(in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    return tmp_3


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 1},  num_warps=1),
        triton.Config({'BLOCK_C': 1},  num_warps=2),
        triton.Config({'BLOCK_C': 2},  num_warps=2),
        triton.Config({'BLOCK_C': 4},  num_warps=4),
        triton.Config({'BLOCK_C': 8},  num_warps=4),
        triton.Config({'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 16}, num_warps=8),
    ],
    key=['C', 'BLOCK_HW'],   # BLOCK_HW is a constexpr; autotune tunes per unique HW
)
@triton.jit
def _sigmoid_scale_kernel(
    in1_ptr, in2_ptr, out_ptr,
    C, HW,
    BLOCK_C:  tl.constexpr,
    BLOCK_HW: tl.constexpr,   # next_power_of_2(HW), always >= HW
):
    # pid → (batch, channel-block)
    pid = tl.program_id(0)
    num_c_blocks = tl.cdiv(C, BLOCK_C)
    b      = pid // num_c_blocks
    c_base = (pid % num_c_blocks) * BLOCK_C

    # Unroll over BLOCK_C channels; load one sigmoid scale per channel
    for c_local in range(BLOCK_C):   # constexpr loop → fully unrolled at JIT time
        c = c_base + c_local
        if c < C:                     # compile-time bound when C is constexpr
            # Load sigmoid scale for this channel
            raw   = tl.load(in2_ptr + c).to(tl.float32)
            scale = tl.sigmoid(raw)

            base    = (b * C + c) * HW
            hw_offs = tl.arange(0, BLOCK_HW)
            mask    = hw_offs < HW
            x1  = tl.load(in1_ptr + base + hw_offs, mask=mask, other=0.0).to(tl.float32)
            out = scale * x1
            tl.store(out_ptr + base + hw_offs, out, mask=mask)


@torch.fx.wrap
def fused_sigmoid_scale(in_1, in_2):
    B  = in_1.shape[0]
    C  = in_1.shape[1]
    HW = in_1.shape[2] * in_1.shape[3]
    # Make HW a constexpr in the compiled kernel: pad to next power-of-2
    BLOCK_HW = triton.next_power_of_2(HW)

    out = torch.empty_like(in_1)

    grid = lambda meta: (B * triton.cdiv(C, meta['BLOCK_C']),)
    _sigmoid_scale_kernel[grid](
        in_1, in_2, out,
        C, HW,
        BLOCK_HW=BLOCK_HW,
    )
    return out


def replacement_func():
    return fused_sigmoid_scale