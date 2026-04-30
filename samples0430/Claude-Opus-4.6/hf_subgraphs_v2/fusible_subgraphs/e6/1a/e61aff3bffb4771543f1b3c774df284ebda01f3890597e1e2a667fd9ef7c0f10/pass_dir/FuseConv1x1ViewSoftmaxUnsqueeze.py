import torch
import triton
import triton.language as tl
from pass_dir.pattern_helper import pattern_gm

pattern = pattern_gm


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_conv1x1_softmax_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)

    # Load bias
    bias = tl.load(bias_ptr).to(tl.float32)

    m_offs = tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    # Initialize accumulator with bias
    acc = tl.full([BLOCK_M], bias, dtype=tl.float32)

    input_base = pid_b * C * M

    # Compute dot product over channels for each spatial position
    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C

        # Load weight block [BLOCK_C]
        w = tl.load(weight_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)

        # Load input[b, c_offs, m_offs]: [BLOCK_C, BLOCK_M]
        inp_ptrs = input_ptr + (input_base + c_offs[:, None] * M + m_offs[None, :])
        inp = tl.load(inp_ptrs, mask=c_mask[:, None] & m_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate weighted sum
        acc += tl.sum(w[:, None] * inp, axis=0)

    # Softmax computation
    max_val = tl.max(tl.where(m_mask, acc, float('-inf')), axis=0)
    exp_val = tl.exp(acc - max_val)
    exp_val = tl.where(m_mask, exp_val, 0.0)
    sum_exp = tl.sum(exp_val, axis=0)
    softmax_val = exp_val / sum_exp

    # Store output
    out_ptrs = output_ptr + pid_b * M + m_offs
    tl.store(out_ptrs, softmax_val, mask=m_mask)


@torch.fx.wrap
def fused_conv1x1_softmax(in_0, in_1, in_2):
    B = in_2.shape[0]
    C = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    M = H * W

    BLOCK_M = triton.next_power_of_2(M)

    # Choose BLOCK_C based on BLOCK_M to balance register pressure
    if BLOCK_M <= 64:
        BLOCK_C = 64
    elif BLOCK_M <= 256:
        BLOCK_C = 32
    elif BLOCK_M <= 1024:
        BLOCK_C = 16
    else:
        BLOCK_C = 8

    BLOCK_C = min(BLOCK_C, triton.next_power_of_2(C))

    # Choose num_warps based on problem size
    if BLOCK_M <= 128:
        num_warps = 2
    elif BLOCK_M <= 512:
        num_warps = 4
    else:
        num_warps = 8

    output = torch.empty((B, 1, M, 1), device=in_2.device, dtype=in_2.dtype)

    grid = (B,)
    fused_conv1x1_softmax_kernel[grid](
        in_2,
        in_1,
        in_0,
        output,
        C, M,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=2,
    )

    return output


def replacement_func():
    return fused_conv1x1_softmax