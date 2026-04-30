import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_10 = tmp_6.contiguous()
    return tmp_10


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.jit
def fused_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in float32: (BLOCK_N) - 1D for M=1 case
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load input values: (BLOCK_K) - 1D
        input_vals = tl.load(
            input_ptr + offs_k,
            mask=offs_k < K,
            other=0.0,
        ).to(tl.float32)

        # Load weight block: (BLOCK_N, BLOCK_K) - row-major, memory-coalesced
        weight_vals = tl.load(
            weight_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        # Element-wise multiply + reduce: broadcast input over N dimension
        # input_vals[None, :] broadcasts to (BLOCK_N, BLOCK_K)
        # sum over axis=1 gives (BLOCK_N)
        acc += tl.sum(input_vals[None, :] * weight_vals, axis=1)

    # Add bias: (BLOCK_N)
    bias_vals = tl.load(
        bias_ptr + offs_n,
        mask=offs_n < N,
        other=0.0,
    ).to(tl.float32)
    acc += bias_vals

    # Store output
    tl.store(
        output_ptr + offs_n,
        acc,
        mask=offs_n < N,
    )


@torch.fx.wrap
def fused_linear_view_transpose(input, weight, bias):
    # Move weight and bias to GPU if needed
    device = input.device
    if weight.device != device:
        weight = weight.to(device)
    if bias is not None and bias.device != device:
        bias = bias.to(device)

    # Ensure contiguous memory layout
    if not input.is_contiguous():
        input = input.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    N = weight.shape[0]  # Output features: 512
    K = weight.shape[1]  # Input features: 512

    # Output shape after view(1,1,-1,64) + transpose(1,2) + contiguous:
    # (1, num_heads, seq_len, head_dim) = (1, 8, 1, 64)
    num_heads = N // 64
    head_dim = 64

    # Get seq_len from input shape
    if input.dim() == 3:
        batch = input.shape[0]
        seq_len = input.shape[1]
    elif input.dim() == 2:
        batch = 1
        seq_len = input.shape[0]
    else:
        batch = 1
        seq_len = 1

    output = torch.empty((batch, num_heads, seq_len, head_dim), dtype=input.dtype, device=device)

    BLOCK_N = 64
    BLOCK_K = 64
    grid = (triton.cdiv(N, BLOCK_N),)

    fused_linear_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N=N,
        K=K,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output


def replacement_func():
    return fused_linear_view_transpose