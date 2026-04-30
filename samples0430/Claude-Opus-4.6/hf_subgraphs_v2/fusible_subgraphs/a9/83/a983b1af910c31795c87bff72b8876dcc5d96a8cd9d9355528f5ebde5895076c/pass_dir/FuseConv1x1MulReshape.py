import torch
import triton
import triton.language as tl


def pattern(bias, weight, input_tensor):
    conv2d = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4


def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)


@triton.jit
def conv1x1_reshape_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C_in,
    HW,
    stride_input_b,
    stride_input_c,
    C_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < C_out  # C_out=17 < BLOCK_M=32

    # Single iteration: K=256, BLOCK_K=256
    k_offsets = tl.arange(0, BLOCK_K)

    # Load weight tile [BLOCK_M, BLOCK_K]
    w_ptrs = weight_ptr + m_offsets[:, None] * C_in + k_offsets[None, :]
    w = tl.load(w_ptrs, mask=m_mask[:, None], other=0.0)

    # Load input tile [BLOCK_K, BLOCK_N]
    i_ptrs = input_ptr + pid_b * stride_input_b + k_offsets[:, None] * stride_input_c + n_offsets[None, :]
    inp = tl.load(i_ptrs)

    # Matrix multiply
    acc = tl.dot(w, inp)

    # Add bias
    bias_vals = tl.load(bias_ptr + m_offsets, mask=m_mask, other=0.0)
    acc += bias_vals[:, None]

    # Store output
    out_ptrs = output_ptr + pid_b * C_out * HW + m_offsets[:, None] * HW + n_offsets[None, :]
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=m_mask[:, None])


@triton.jit
def conv1x1_reshape_kernel_loop(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C_in,
    HW,
    stride_input_b,
    stride_input_c,
    C_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_offsets < C_out

    # Loop over K dimension
    for k_start in range(0, C_in, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load weight tile [BLOCK_M, BLOCK_K] - reused across blocks, keep in cache
        w_ptrs = weight_ptr + m_offsets[:, None] * C_in + k_offsets[None, :]
        w = tl.load(w_ptrs, mask=m_mask[:, None], other=0.0, eviction_policy='evict_last')

        # Load input tile [BLOCK_K, BLOCK_N] - used once per block, evict early
        i_ptrs = input_ptr + pid_b * stride_input_b + k_offsets[:, None] * stride_input_c + n_offsets[None, :]
        inp = tl.load(i_ptrs, eviction_policy='evict_first')

        # Matrix multiply accumulate
        acc += tl.dot(w, inp)

    # Add bias
    bias_vals = tl.load(bias_ptr + m_offsets, mask=m_mask, other=0.0)
    acc += bias_vals[:, None]

    # Store output
    out_ptrs = output_ptr + pid_b * C_out * HW + m_offsets[:, None] * HW + n_offsets[None, :]
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=m_mask[:, None])


@torch.fx.wrap
def conv1x1_reshape_func(bias, weight, input_tensor):
    B = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    HW = H * W  # 4096
    C_out = weight.shape[0]  # 17

    # Allocate output directly in final shape [B, C_out, HW] = [B, 17, 4096]
    output = torch.empty((B, C_out, HW), dtype=input_tensor.dtype, device=input_tensor.device)

    BLOCK_M = 32  # Next power of 2 >= C_out (17)
    BLOCK_N = 128

    elem_size = input_tensor.element_size()

    if elem_size <= 2:
        # fp16/bf16: pipelined loop with BLOCK_K=64, 4 iterations, 3-stage pipeline
        BLOCK_K = 64
        grid = (HW // BLOCK_N, B)
        conv1x1_reshape_kernel_loop[grid](
            input_tensor, weight, bias, output,
            C_in, HW,
            input_tensor.stride(0), input_tensor.stride(1),
            C_out=C_out, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
    else:
        # fp32: loop kernel with BLOCK_K=128, pipelined (2 iterations)
        BLOCK_K = 128
        grid = (HW // BLOCK_N, B)
        conv1x1_reshape_kernel_loop[grid](
            input_tensor, weight, bias, output,
            C_in, HW,
            input_tensor.stride(0), input_tensor.stride(1),
            C_out=C_out, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

    return output


def replacement_func():
    return conv1x1_reshape_func