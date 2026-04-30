import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim=2)
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    tmp_9 = tmp_5.unsqueeze(3)
    tmp_10 = tmp_8 - tmp_6
    return (tmp_10, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_sq_dist_softmax_kernel(
    in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    C: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused kernel: (in_1 - in_2)^2 -> sum(dim=3) -> * scale -> softmax(dim=2) -> output"""
    pid_n = tl.program_id(0)
    c_range = tl.arange(0, C)  # [32]
    acc = tl.zeros([C], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # in_1[0, pid_n, c, k] at offset pid_n*C*K + c*K + k
        in_1_offsets = pid_n * C * K + c_range[:, None] * K + k_range[None, :]
        in_1_vals = tl.load(in_1_ptr + in_1_offsets).to(tl.float32)
        # in_2[0, 0, c, k] at offset c*K + k
        in_2_offsets = c_range[:, None] * K + k_range[None, :]
        in_2_vals = tl.load(in_2_ptr + in_2_offsets).to(tl.float32)
        diff = in_1_vals - in_2_vals
        acc += tl.sum(diff * diff, axis=1)

    # Multiply by scale: in_3[0, 0, c] at offset c
    scale = tl.load(in_3_ptr + c_range).to(tl.float32)
    acc = acc * scale

    # Softmax over C=32 elements
    max_val = tl.max(acc, axis=0)
    exp_val = tl.exp(acc - max_val)
    sum_exp = tl.sum(exp_val, axis=0)
    softmax_val = exp_val / sum_exp

    # Store: out[0, n, c, 0] at offset n*C + c
    tl.store(out_ptr + pid_n * C + c_range, softmax_val.to(out_ptr.dtype.element_ty))


@triton.jit
def broadcast_sub_kernel(
    in_4_ptr, in_0_ptr, out_ptr,
    C: tl.constexpr, K: tl.constexpr,
):
    """Compute out[n, c, k] = in_4[n, k] - in_0[c, k]"""
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    k_range = tl.arange(0, K)  # [512]

    # in_4[0, n, k] at offset n*K + k
    in_4_vals = tl.load(in_4_ptr + pid_n * K + k_range)
    # in_0[c, k] at offset c*K + k
    in_0_vals = tl.load(in_0_ptr + pid_c * K + k_range)
    result = in_4_vals - in_0_vals

    # out[0, n, c, k] at offset n*C*K + c*K + k
    tl.store(out_ptr + pid_n * C * K + pid_c * K + k_range, result)


@torch.fx.wrap
def fused_encnet_kernel(in_0, in_1, in_2, in_3, in_4):
    N = 4096
    C = 32
    K = 512
    dtype = in_1.dtype
    device = in_1.device

    # Compute tmp_9: fused squared distance + softmax + unsqueeze [1, N, C, 1]
    tmp_9 = torch.empty((1, N, C, 1), dtype=dtype, device=device)
    fused_sq_dist_softmax_kernel[(N,)](
        in_1, in_2, in_3, tmp_9,
        C=C, K=K, BLOCK_K=128,
        num_warps=4,
    )

    # Compute tmp_10: broadcast subtract [1, N, C, K]
    tmp_10 = torch.empty((1, N, C, K), dtype=dtype, device=device)
    broadcast_sub_kernel[(N, C)](
        in_4, in_0, tmp_10,
        C=C, K=K,
        num_warps=4,
    )

    return (tmp_10, tmp_9)


def replacement_func():
    return fused_encnet_kernel