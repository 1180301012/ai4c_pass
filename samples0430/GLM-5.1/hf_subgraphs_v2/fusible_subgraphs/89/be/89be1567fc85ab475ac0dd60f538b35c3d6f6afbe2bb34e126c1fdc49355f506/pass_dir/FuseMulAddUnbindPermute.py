import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_mul_add_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr, out_ptr,
    K: tl.constexpr,
    s_in2_0, s_in2_1, s_in2_3,
    s_in1_2, s_in1_3,
    s_in0_0, s_in0_1,
    s_out_0, s_out_1, s_out_2, s_out_3,
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)

    k = tl.arange(0, K)

    # Load in_2[b, i, 0, k] - j=0 always since in_2's dim 2 has size 1
    in_2_val = tl.load(in_2_ptr + pid_b * s_in2_0 + pid_i * s_in2_1 + k * s_in2_3)

    # Load in_1[0, 0, 0, k] and in_1[0, 0, 1, k]
    # First two dims are always 0 (shape [1, 1, 2, 128])
    in1_val0 = tl.load(in_1_ptr + k * s_in1_3)
    in1_val1 = tl.load(in_1_ptr + s_in1_2 + k * s_in1_3)

    # Load in_0[0, k] and in_0[1, k]
    # in_0 has shape [2, 128]
    in0_val0 = tl.load(in_0_ptr + k * s_in0_1)
    in0_val1 = tl.load(in_0_ptr + s_in0_0 + k * s_in0_1)

    # Fused multiply-add for both slices (j=0 and j=1)
    val0 = in_2_val * in1_val0 + in0_val0  # j=0
    val1 = in_2_val * in1_val1 + in0_val1  # j=1

    # Store out[b, i, 0, k] - first slice (j=0)
    tl.store(out_ptr + pid_b * s_out_0 + pid_i * s_out_1 + k * s_out_3, val0)

    # Store out[b, i, 1, k] - second slice (j=1)
    tl.store(out_ptr + pid_b * s_out_0 + pid_i * s_out_1 + s_out_2 + k * s_out_3, val1)


@torch.fx.wrap
def fused_mul_add(in_0, in_1, in_2):
    B = in_2.shape[0]
    I = in_2.shape[1]   # Always 17
    K = in_2.shape[3]   # Always 128
    J = in_1.shape[2]   # Always 2

    # Output shape: [B, I, J, K] = [B, 17, 2, 128]
    out = torch.empty((B, I, J, K), dtype=in_2.dtype, device=in_2.device)

    # 2D grid: (batch_size, I_dim=17)
    grid = (B, I)

    fused_mul_add_kernel[grid](
        in_2, in_1, in_0, out,
        K=K,
        s_in2_0=in_2.stride(0), s_in2_1=in_2.stride(1), s_in2_3=in_2.stride(3),
        s_in1_2=in_1.stride(2), s_in1_3=in_1.stride(3),
        s_in0_0=in_0.stride(0), s_in0_1=in_0.stride(1),
        s_out_0=out.stride(0), s_out_1=out.stride(1), s_out_2=out.stride(2), s_out_3=out.stride(3),
    )

    return out


def replacement_func():
    return fused_mul_add