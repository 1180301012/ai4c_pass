import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(8, -1)
    tmp_2 = tmp_1.view(8, -1, 1, 1)
    tmp_3 = tmp_2.view(8, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _softmax_weighted_sum_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, C, HW,
    in0_s0, in0_s1, in0_s2,
    in1_s0, in1_s1,
    out_s0, out_s1, out_s2,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (b, c) pair and one BLOCK_SIZE chunk of HW
    pid = tl.program_id(0)
    num_hw_blocks = tl.cdiv(HW, BLOCK_SIZE)
    bc_pid       = pid // num_hw_blocks
    hw_block_pid = pid  % num_hw_blocks

    b = bc_pid // C
    c = bc_pid  % C

    hw_start   = hw_block_pid * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask    = hw_offsets < HW

    # ------------------------------------------------------------------ #
    # Compute softmax weights from in_1 shape [B, 2, 1, C]
    # We only need the 2 scalar weights for (b, c).
    # ------------------------------------------------------------------ #
    in1_base = b * in1_s0 + c * in1_s1
    k0 = tl.load(in1_ptr + in1_base,                    mask=hw_mask, other=0.0)
    k1 = tl.load(in1_ptr + in1_base + in1_s1,           mask=hw_mask, other=0.0)

    # Promote to float32 for numerically stable softmax
    k0_f32 = k0.to(tl.float32)
    k1_f32 = k1.to(tl.float32)
    max_k  = tl.maximum(k0_f32, k1_f32)
    e0     = tl.exp(k0_f32 - max_k)
    e1     = tl.exp(k1_f32 - max_k)
    inv_s  = 1.0 / (e0 + e1)
    w0     = e0 * inv_s          # float32 weight for branch k=0
    w1     = e1 * inv_s          # float32 weight for branch k=1

    # ------------------------------------------------------------------ #
    # Load in_0[b, 0, c, :] and in_0[b, 1, c, :] and compute weighted sum
    # in_0 shape [B, 2, C, H, W]  (contiguous → s2 = HW, s3 = W, s4 = 1)
    # ------------------------------------------------------------------ #
    in0_base   = b * in0_s0 + c * in0_s2
    val_k0_f32 = tl.load(in0_ptr + in0_base +       hw_offsets,
                         mask=hw_mask, other=0.0).to(tl.float32)
    val_k1_f32 = tl.load(in0_ptr + in0_base + in0_s1 + hw_offsets,
                         mask=hw_mask, other=0.0).to(tl.float32)

    result_f32 = w0 * val_k0_f32 + w1 * val_k1_f32

    # ------------------------------------------------------------------ #
    # Cast back to input dtype and store into out[b, c, :]
    # ------------------------------------------------------------------ #
    out_base = b * out_s0 + c * out_s1
    if IS_FP16:
        tl.store(out_ptr + out_base + hw_offsets, result_f32.to(tl.float16),   mask=hw_mask)
    elif IS_BF16:
        tl.store(out_ptr + out_base + hw_offsets, result_f32.to(tl.bfloat16),  mask=hw_mask)
    else:
        tl.store(out_ptr + out_base + hw_offsets, result_f32,                   mask=hw_mask)


@torch.fx.wrap
def softmax_weighted_sum(in_0, in_1):
    B   = in_0.shape[0]
    C   = in_0.shape[2]
    HW  = in_0.shape[3] * in_0.shape[4]

    out = torch.empty((B, C, in_0.shape[3], in_0.shape[4]),
                      dtype=in_0.dtype, device=in_0.device)

    IS_FP16 = (in_0.dtype == torch.float16)
    IS_BF16 = (in_0.dtype == torch.bfloat16)

    grid = lambda meta: (B * C * triton.cdiv(HW, meta['BLOCK_SIZE']),)

    _softmax_weighted_sum_kernel[grid](
        in_0, in_1, out,
        B, C, HW,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1),
        out.stride(0),  out.stride(1),  out.stride(2),
        IS_FP16=IS_FP16,
        IS_BF16=IS_BF16,
    )
    return out


def replacement_func():
    return softmax_weighted_sum