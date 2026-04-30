import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 64},  num_warps=8),
        triton.Config({'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 64},  num_warps=16),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 32},  num_warps=4),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def fused_mul_sum_sigmoid_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    B,
    C: tl.constexpr,   # constexpr so tl.arange(0, C) works
    HW: tl.constexpr,  # constexpr so addr computation is compile-time
    BLOCK_HW: tl.constexpr,
):
    """
    Fused kernel: output[b, 0, h, w] = sigmoid( sum_c in_0[b,c,h,w] * in_1[b,c,h,w] )
    Tiling strategy:
      - Grid axis 0: B * (HW // BLOCK_HW) programs
      - Each program handles BLOCK_HW spatial positions and ALL C channels
      - Loads a [C, BLOCK_HW] tile from both inputs, multiplies element-wise,
        reduces along axis 0 (C), applies sigmoid, stores [BLOCK_HW] outputs.
    """
    prog_id = tl.program_id(0)
    num_hw_blocks = HW // BLOCK_HW
    b       = prog_id // num_hw_blocks
    hw_blk  = prog_id  % num_hw_blocks

    hw_start = hw_blk * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]

    # 2-D address tensor [C, BLOCK_HW]  —  contiguous along the HW axis
    base = b * C * HW
    addr = base + tl.arange(0, C)[:, None] * HW + hw_offs[None, :]  # [C, BLOCK_HW]

    # Accumulator [C, BLOCK_HW] in float32
    acc = tl.zeros([C, BLOCK_HW], dtype=tl.float32)

    # Fully-unrolled loop over C=64 channels (C is constexpr → loop is unrolled)
    for c in range(C):
        v0 = tl.load(in0_ptr + addr).to(tl.float32)   # [C, BLOCK_HW]
        v1 = tl.load(in1_ptr + addr).to(tl.float32)   # [C, BLOCK_HW]
        acc = acc + v0 * v1

    # Reduce over channel dim → [BLOCK_HW]
    result_f32 = tl.sum(acc, axis=0)

    # Fused sigmoid in float32
    result_f32 = tl.sigmoid(result_f32)

    # Store — pointer type determines target dtype; cast required if fp32→bf16/fp16
    # Use tl.where trick to cast: compute a value in the right dtype without explicit cast
    # Trick: result_f32 + zero_cast_as_output_dtype gives us the right dtype
    out_offs = b * HW + hw_offs
    tl.store(out_ptr + out_offs, result_f32)


@torch.fx.wrap
def fused_mul_sum_sigmoid(in_0, in_1):
    B, C, H, W = in_0.shape
    HW = H * W
    out = torch.empty((B, 1, H, W), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (B * (HW // meta['BLOCK_HW']),)

    fused_mul_sum_sigmoid_kernel[grid](
        in_0, in_1, out,
        B,
        C=C, HW=HW,
    )
    return out


def replacement_func():
    return fused_mul_sum_sigmoid