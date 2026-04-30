import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['HW'],
)
@triton.jit
def _fused_softmax_mul_sum_kernel_b1(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    bc_idx = tl.program_id(0)
    hw_block = tl.program_id(1)

    b = bc_idx // C
    c = bc_idx % C

    # Compute softmax weights (dim=1, size=2) in float32 for stability
    in1_base = b * 2 * C + c
    val_0 = tl.load(in_1_ptr + in1_base).to(tl.float32)
    val_1 = tl.load(in_1_ptr + in1_base + C).to(tl.float32)

    max_val = tl.maximum(val_0, val_1)
    exp_0 = tl.exp(val_0 - max_val)
    exp_1 = tl.exp(val_1 - max_val)
    sum_exp = exp_0 + exp_1
    w0 = exp_0 / sum_exp
    w1 = exp_1 / sum_exp

    # Load spatial data and compute weighted sum
    hw_offsets = hw_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = hw_offsets < HW

    # in_0 is [B, 2, C, H, W] contiguous
    base_0 = b * 2 * C * HW + c * HW
    base_1 = base_0 + C * HW

    x0 = tl.load(in_0_ptr + base_0 + hw_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in_0_ptr + base_1 + hw_offsets, mask=mask, other=0.0).to(tl.float32)

    result = w0 * x0 + w1 * x1

    # Store result - out is [B, C, H, W]
    out_base = b * C * HW + c * HW
    tl.store(out_ptr + out_base + hw_offsets, result, mask=mask)


@torch.fx.wrap
def fused_softmax_weighted_sum_b1(in_0, in_1):
    B = in_0.shape[0]
    C = in_0.shape[2]
    H = in_0.shape[3]
    W = in_0.shape[4]
    HW = H * W

    out = torch.empty((B, C, H, W), device=in_0.device, dtype=in_0.dtype)

    grid = lambda meta: (B * C, triton.cdiv(HW, meta['BLOCK_SIZE']))

    _fused_softmax_mul_sum_kernel_b1[grid](
        in_0, in_1, out,
        C, HW,
    )

    return out


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(1, -1)
    tmp_2 = tmp_1.view(1, -1, 1, 1)
    tmp_3 = tmp_2.view(1, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_softmax_weighted_sum_b1