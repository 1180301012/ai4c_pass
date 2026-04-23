import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 300, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim = -1)
    tmp_3 = tmp_2.view(1, 8, 300, 625)
    tmp_4 = tmp_3.view(8, 300, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p = 0.0, training = False)
    return (tmp_5, tmp_3)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_8_300_625")


@triton.jit
def fused_add_softmax_kernel(
    in0_ptr, in1_ptr, out_ptr,
    N, M, num_heads,
    IS_FLOAT16: tl.constexpr,
    IS_BFLOAT16: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row_idx = tl.program_id(0)
    total_rows = num_heads * N
    if row_idx >= total_rows:
        return

    pos = row_idx % N
    offsets = tl.arange(0, BLOCK_M)
    mask = offsets < M

    # Load inputs and upcast to float32 for numerical stability
    in0 = tl.load(in0_ptr + pos * M + offsets, mask=mask, other=-float('inf')).to(tl.float32)
    in1 = tl.load(in1_ptr + row_idx * M + offsets, mask=mask, other=-float('inf')).to(tl.float32)

    # Add with broadcasting
    x = in0 + in1

    # Softmax: max -> subtract -> exp -> sum -> divide
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    result = exp_x / sum_exp

    # Downcast to original dtype and store
    if IS_FLOAT16:
        output = result.to(tl.float16)
    elif IS_BFLOAT16:
        output = result.to(tl.bfloat16)
    else:
        output = result
    tl.store(out_ptr + row_idx * M + offsets, output, mask=mask)


def _fused_add_softmax(in_0, in_1):
    in_0_c = in_0.contiguous()
    in_1_c = in_1.contiguous()

    N = in_0_c.shape[2]
    M = in_0_c.shape[3]
    num_heads = in_1_c.shape[1]
    num_rows = num_heads * N

    is_float16 = (in_1_c.dtype == torch.float16)
    is_bfloat16 = (in_1_c.dtype == torch.bfloat16)

    output = torch.empty((num_heads, N, M), dtype=in_1_c.dtype, device=in_1_c.device)

    BLOCK_M = 1024

    grid = (num_rows,)
    fused_add_softmax_kernel[grid](
        in0_ptr=in_0_c, in1_ptr=in_1_c, out_ptr=output,
        N=N, M=M, num_heads=num_heads,
        IS_FLOAT16=is_float16,
        IS_BFLOAT16=is_bfloat16,
        BLOCK_M=BLOCK_M,
    )

    # Return (dropout_result, 4d_view) — dropout is identity (p=0.0, training=False)
    return (output, output.view(1, num_heads, N, M))


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, route):
    if route == "route_8_300_625":
        return _fused_add_softmax(in_0, in_1)
    elif route == "route_8_625_625":
        return _fused_add_softmax(in_0, in_1)
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return kernel_wrapper