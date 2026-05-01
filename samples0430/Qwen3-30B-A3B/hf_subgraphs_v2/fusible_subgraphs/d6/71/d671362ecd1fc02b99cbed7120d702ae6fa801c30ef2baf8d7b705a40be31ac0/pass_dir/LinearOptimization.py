import torch
import triton
import triton.language as tl

@triton.jit
def _linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    weight_stride0,
    weight_stride1,
    bias_stride0,
    output_stride0,
    output_stride1,
    output_stride2,
    B,
    S,
    I,
    O,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    m_start = m_block * BLOCK_SIZE_M
    n_start = n_block * BLOCK_SIZE_N
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, I, BLOCK_SIZE_K):
        input_block = tl.load(
            input_ptr + m_start * input_stride0 + k * input_stride2,
            shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            mask=(m_start + tl.arange(0, BLOCK_SIZE_M) < B * S, k + tl.arange(0, BLOCK_SIZE_K) < I),
            other=0.0
        )
        weight_block = tl.load(
            weight_ptr + k * weight_stride0 + n_start * weight_stride1,
            shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            mask=(k + tl.arange(0, BLOCK_SIZE_K) < I, n_start + tl.arange(0, BLOCK_SIZE_N) < O),
            other=0.0
        )
        acc += tl.dot(input_block, weight_block)
    if bias_ptr is not None:
        bias_block = tl.load(
            bias_ptr + n_start,
            shape=(BLOCK_SIZE_N, ),
            mask=n_start + tl.arange(0, BLOCK_SIZE_N) < O,
            other=0.0
        )
        acc += bias_block
    tl.store(
        output_ptr + m_start * output_stride0 + n_start * output_stride2,
        acc,
        mask=(m_start + tl.arange(0, BLOCK_SIZE_M) < B * S, n_start + tl.arange(0, BLOCK_SIZE_N) < O)
    )

@torch.fx.wrap
def optimized_linear(input, weight, bias):
    B, S, I = input.shape
    O = weight.shape[0]
    output = torch.empty((B, S, O), dtype=input.dtype, device=input.device)
    input_stride0, input_stride1, input_stride2 = input.stride()
    weight_stride0, weight_stride1 = weight.stride()
    bias_stride0 = bias.stride()[0] if bias is not None else 0
    output_stride0, output_stride1, output_stride2 = output.stride()
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    grid_m = (B * S + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (O + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    _linear_kernel[(grid_m, grid_n)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        input_stride0=input_stride0,
        input_stride1=input_stride1,
        input_stride2=input_stride2,
        weight_stride0=weight_stride0,
        weight_stride1=weight_stride1,
        bias_stride0=bias_stride0,
        output_stride0=output_stride0,
        output_stride1=output_stride1,
        output_stride2=output_stride2,
        B=B,
        S=S,
        I=I,
        O=O,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return output

def pattern(in_3, in_2, in_1):
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    return linear

def replacement_args(in_3, in_2, in_1):
    return (in_3, in_2, in_1)

def replacement_func():
    return optimized_linear