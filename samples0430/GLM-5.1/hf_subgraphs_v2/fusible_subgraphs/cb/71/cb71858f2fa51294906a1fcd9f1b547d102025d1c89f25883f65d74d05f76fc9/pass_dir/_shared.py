import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv2d_view_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    N_IN: tl.constexpr,
    N_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    m_start = pid * BLOCK_M
    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < N_OUT

    n_range = tl.arange(0, N_IN)

    # Load input vector (flat, contiguous)
    x = tl.load(input_ptr + n_range).to(tl.float32)

    # Load bias for this block of output channels
    b = tl.load(bias_ptr + m_range, mask=m_mask, other=0.0).to(tl.float32)

    # Load weight block: weight[m, n] = weight_ptr + m * N_IN + n
    w_ptrs = weight_ptr + m_range[:, None] * N_IN + n_range[None, :]
    w = tl.load(w_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    # Compute dot product: sum(weight * input) over input dimension
    acc = tl.sum(w * x[None, :], axis=1)

    # Add bias and apply sigmoid
    result = tl.sigmoid(acc + b)

    # Store output (flat, contiguous - will be viewed as [1, 2, 8, 8])
    tl.store(output_ptr + m_range, result, mask=m_mask)


@triton.jit
def fused_sum_div_dim3_kernel(
    input_ptr, output_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_ROWS: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < ROW_SIZE

    # Process all rows in a single program (loop unrolled by Triton)
    for row_idx in tl.range(0, N_ROWS):
        row_start = row_idx * ROW_SIZE

        # Load row elements
        x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)

        # Compute sum
        row_sum = tl.sum(x, axis=0)

        # Divide each element by sum
        out = x / row_sum

        # Store result
        tl.store(output_ptr + row_start + offsets, out, mask=mask)


def _fused_conv2d_view_sigmoid(input, weight, bias):
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N_IN = input.numel()  # 16 = 2 * 1 * 8
    N_OUT = weight.shape[0]  # 128
    BLOCK_M = 128  # Single program for minimal launch overhead

    output = torch.empty(1, 2, 8, 8, dtype=input.dtype, device=input.device)

    grid = (1,)

    fused_conv2d_view_sigmoid_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        N_IN=N_IN,
        N_OUT=N_OUT,
        BLOCK_M=BLOCK_M,
        num_warps=4,
        num_stages=2,
    )

    return output


def _fused_sum_div_dim3(input_tensor):
    input_tensor = input_tensor.contiguous()

    ROW_SIZE = 8
    BLOCK_SIZE = 8
    N_ROWS = 16  # 1 * 2 * 8

    output = torch.empty_like(input_tensor)

    grid = (1,)

    fused_sum_div_dim3_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        ROW_SIZE=ROW_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        N_ROWS=N_ROWS,
        num_warps=1,
        num_stages=2,
    )

    return output


@torch.fx.wrap
def shared_dispatch_wrapper(*args):
    route = args[-1]
    if route == "conv2d_view_sigmoid":
        input, weight, bias = args[0], args[1], args[2]
        return _fused_conv2d_view_sigmoid(input, weight, bias)
    elif route == "sum_div_dim3":
        input_tensor = args[0]
        return _fused_sum_div_dim3(input_tensor)
    elif route == "full_model":
        input_conv, weight, bias, in_3 = args[0], args[1], args[2], args[3]
        tmp_4 = _fused_conv2d_view_sigmoid(input_conv, weight, bias)
        tmp_6 = _fused_sum_div_dim3(in_3)
        return (tmp_6, tmp_4)
    else:
        raise ValueError(f"Unknown route: {route}")