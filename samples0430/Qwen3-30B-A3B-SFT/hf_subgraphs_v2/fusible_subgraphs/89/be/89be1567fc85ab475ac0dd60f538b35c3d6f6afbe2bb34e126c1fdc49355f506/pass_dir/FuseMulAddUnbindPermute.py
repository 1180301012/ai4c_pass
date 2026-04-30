import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pass 1 of 2: fuse  tmp_1 = in_2 * in_1;  tmp_2 = tmp_1 + in_0
# Returns the [B, T, 2, C] tensor.  No unbind in this pattern.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def _fused_mul_add_2d_kernel(
    in0_ptr,   # [2, C]
    in1_ptr,   # [1, 1, 2, C]
    in2_ptr,   # [B, T, 1, C]
    out_ptr,   # [B, T, 2, C]
    T,
    C: tl.constexpr,
):
    b_idx = tl.program_id(0)
    t_idx = tl.program_id(1)
    k = tl.arange(0, C)

    in2_val = tl.load(in2_ptr + b_idx * T * C + t_idx * C + k)
    in1_0   = tl.load(in1_ptr + k)
    in1_1   = tl.load(in1_ptr + C + k)
    in0_0   = tl.load(in0_ptr + k)
    in0_1   = tl.load(in0_ptr + C + k)

    tmp0 = in2_val * in1_0 + in0_0
    tmp1 = in2_val * in1_1 + in0_1

    out_base = out_ptr + b_idx * T * 2 * C + t_idx * 2 * C
    tl.store(out_base + k,       tmp0)
    tl.store(out_base + C + k,   tmp1)


@torch.fx.wrap
def _fused_mul_add(in_0, in_1, in_2):
    B = in_2.shape[0]
    T = in_2.shape[1]
    C = 128
    out = torch.empty((B, T, 2, C), dtype=in_2.dtype, device=in_2.device)
    _fused_mul_add_2d_kernel[(B, T)](in_0, in_1, in_2, out, T, C, num_warps=4)
    return out


def replacement_func():
    return _fused_mul_add