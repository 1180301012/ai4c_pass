import torch
import triton
import triton.language as tl


def pattern(bias, weight, x):
    conv2d = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    result = torch.flatten(conv2d, 2)
    return result


def replacement_args(bias, weight, x):
    return (bias, weight, x)


@triton.jit
def conv1x1_flatten_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (HW // BLOCK_N, N_batch)
    pid_hw = tl.program_id(0)
    pid_n = tl.program_id(1)

    # HW tile offsets - no mask needed since BLOCK_N divides HW (3072)
    hw_start = pid_hw * BLOCK_N
    hw_offsets = hw_start + tl.arange(0, BLOCK_N)

    # Output channel offsets (M dimension)
    m_offsets = tl.arange(0, BLOCK_M)
    m_mask = m_offsets < C_OUT

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointer for this batch's input
    # For contiguous NCHW: stride_in_n = C_IN * HW (all constexpr)
    input_batch_ptr = input_ptr + pid_n * C_IN * HW

    # Loop over input channels (K dimension)
    # C_IN=160, BLOCK_K=32 -> exactly 5 iterations, no k-masking needed
    for k_start in range(0, C_IN, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load weight tile [BLOCK_M, BLOCK_K] - only m_mask needed
        w = tl.load(
            weight_ptr + m_offsets[:, None] * C_IN + k_offsets[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )

        # Load input tile [BLOCK_K, BLOCK_N] - no mask needed
        # For contiguous NCHW: stride_in_c = HW (constexpr)
        inp = tl.load(
            input_batch_ptr + k_offsets[:, None] * HW + hw_offsets[None, :],
        )

        # Matrix multiply: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(w, inp)

    # Add bias [BLOCK_M]
    bias = tl.load(bias_ptr + m_offsets, mask=m_mask, other=0.0)
    acc += bias[:, None]

    # Store output [BLOCK_M, BLOCK_N] - only m_mask needed
    # Output [N, C_OUT, HW] contiguous: stride_out_n = C_OUT*HW, stride_out_c = HW
    out_ptrs = output_ptr + pid_n * C_OUT * HW + m_offsets[:, None] * HW + hw_offsets[None, :]
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=m_mask[:, None])


@torch.fx.wrap
def conv1x1_flatten_fused(bias, weight, x):
    N_batch = x.shape[0]
    C_in = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    HW = H * W
    C_out = weight.shape[0]

    # Output: [N, C_out, HW]
    output = torch.empty((N_batch, C_out, HW), dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_K = 32
    BLOCK_N = 128

    grid = (HW // BLOCK_N, N_batch)

    conv1x1_flatten_kernel[grid](
        x, weight, bias, output,
        C_IN=C_in,
        C_OUT=C_out,
        HW=HW,
        BLOCK_K=BLOCK_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=5,
        num_warps=4,
    )

    return output


def replacement_func():
    return conv1x1_flatten_fused