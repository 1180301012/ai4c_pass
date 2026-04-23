import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    in_2 = in_1 + in_0
    tmp_1 = in_2.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(in_2)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N', 'dtype_id'],
)
@triton.jit
def fused_add_softmax_kernel(
    in0_ptr, in1_ptr, out_ptr,
    N, n_rows,
    BLOCK_SIZE: tl.constexpr,
    dtype_id: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = row_idx * N
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load inputs and cast to float32
    in0 = tl.load(in0_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    in1 = tl.load(in1_ptr + row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Add
    x = in0 + in1

    # Softmax: subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max

    # Exp
    x_exp = tl.exp(x_shifted)

    # Sum
    x_sum = tl.sum(x_exp, axis=0)

    # Normalize
    result = x_exp / x_sum

    # Cast back to original dtype
    if dtype_id == 0:
        result = result.to(tl.float16)
    elif dtype_id == 1:
        pass  # already float32
    elif dtype_id == 2:
        result = result.to(tl.bfloat16)

    tl.store(out_ptr + row_start + col_offsets, result, mask=mask)


@torch.fx.wrap
def fused_add_softmax_dropout(in_0, in_1):
    shape = in_0.shape
    dtype = in_0.dtype

    N = shape[-1]  # last dimension size (softmax dimension)
    n_rows = in_0.numel() // N  # total number of rows

    # Determine dtype_id
    if dtype == torch.float16:
        dtype_id = 0
    elif dtype == torch.float32:
        dtype_id = 1
    elif dtype == torch.bfloat16:
        dtype_id = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    out = torch.empty_like(in_0)

    grid = (triton.cdiv(n_rows, 1),)

    fused_add_softmax_kernel[grid](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        N=N,
        n_rows=n_rows,
        BLOCK_SIZE=max(16, triton.next_power_of_2(N)),
        dtype_id=dtype_id,
    )

    return out


def replacement_func():
    return fused_add_softmax_dropout