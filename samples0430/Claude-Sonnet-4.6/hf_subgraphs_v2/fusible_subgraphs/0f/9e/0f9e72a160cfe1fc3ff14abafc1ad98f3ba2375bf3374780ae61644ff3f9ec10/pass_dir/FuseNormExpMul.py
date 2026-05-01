import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return (tmp_6, tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def _fused_norm_exp_mul_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out2_ptr,
    out4_ptr,
    out6_ptr,
    D: tl.constexpr,
):
    """
    Single program handles ALL work for one batch row of both in1 and in2.
    Computes:
      tmp2 = in1 / ||in1||_2    (L2-normalize in1 along last dim)
      tmp4 = in2 / ||in2||_2    (L2-normalize in2 along last dim)
      tmp6 = exp(in0) * tmp4    (scale normalized in2 by exp(scalar))
    No reshape: contiguous memory layout assumed (valid for [1,512] and [1,1,512]).
    """
    row_id = tl.program_id(0)
    offsets = tl.arange(0, D)

    # ---- Normalize in_1[row_id] -> tmp2 ----
    x1 = tl.load(in1_ptr + row_id * D + offsets)
    x1_f32 = x1.to(tl.float32)
    norm1 = tl.sqrt(tl.sum(x1_f32 * x1_f32, axis=0))
    tmp2 = (x1_f32 / norm1).to(x1.dtype)
    tl.store(out2_ptr + row_id * D + offsets, tmp2)

    # ---- Normalize in_2[row_id] -> tmp4 ----
    x2 = tl.load(in2_ptr + row_id * D + offsets)
    x2_f32 = x2.to(tl.float32)
    norm2 = tl.sqrt(tl.sum(x2_f32 * x2_f32, axis=0))
    tmp4_f32 = x2_f32 / norm2
    tmp4 = tmp4_f32.to(x2.dtype)
    tl.store(out4_ptr + row_id * D + offsets, tmp4)

    # ---- Compute tmp6 = exp(in_0) * tmp4 (no extra memory round-trip) ----
    in0_val = tl.load(in0_ptr).to(tl.float32)
    scale = tl.exp(in0_val)
    tmp6 = (tmp4_f32 * scale).to(x2.dtype)
    tl.store(out6_ptr + row_id * D + offsets, tmp6)


@torch.fx.wrap
def fused_norm_exp_mul(in_0, in_1, in_2):
    """
    Fused kernel: computes the full 6-op subgraph in ONE kernel launch.
      tmp2 = in_1 / ||in_1||_2
      tmp4 = in_2 / ||in_2||_2
      tmp6 = exp(in_0) * tmp4
    Returns (tmp6, tmp4, tmp2) — same order as model.py return.
    No reshape/view calls (blocked API); uses contiguous memory layout directly.
    """
    D = in_1.shape[-1]
    # B = number of rows; for [1,512] -> B=1, for [1,1,512] -> B=1
    B = in_1.numel() // D

    out2 = torch.empty_like(in_1)   # [1, 512]
    out4 = torch.empty_like(in_2)   # [1, 1, 512]
    out6 = torch.empty_like(in_2)   # [1, 1, 512]

    _fused_norm_exp_mul_kernel[(B,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        out2_ptr=out2,
        out4_ptr=out4,
        out6_ptr=out6,
        D=D,
    )

    return (out6, out4, out2)


def replacement_func():
    return fused_norm_exp_mul