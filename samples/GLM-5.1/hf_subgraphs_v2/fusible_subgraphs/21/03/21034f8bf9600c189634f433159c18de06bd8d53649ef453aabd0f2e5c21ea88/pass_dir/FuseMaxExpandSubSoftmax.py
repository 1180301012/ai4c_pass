import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    B,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused softmax kernel: max + sub + exp + sum + div in one kernel."""
    row_idx = tl.program_id(0)
    total_rows = B * M
    if row_idx >= total_rows:
        return

    # For a [B, M, N] row-major tensor, row_start = row_idx * N
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load the row
    row = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)

    # Numerical stability: subtract max before exp
    row_max = tl.max(row, axis=0)

    # Subtract max and compute exp
    safe_row = row - row_max
    exp_row = tl.exp(safe_row)

    # Sum exp values for normalization
    sum_exp = tl.sum(exp_row, axis=0)

    # Normalize: softmax = exp(x - max) / sum(exp(x - max))
    result = exp_row / sum_exp

    # Store the result
    tl.store(output_ptr + row_start + offsets, result, mask=mask)


@torch.fx.wrap
def fused_softmax(in_0):
    B, M, N = in_0.shape
    output = torch.empty_like(in_0)
    BLOCK_SIZE = triton.next_power_of_2(N)
    total_rows = B * M
    fused_softmax_kernel[(total_rows,)](
        input_ptr=in_0,
        output_ptr=output,
        B=B,
        M=M,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def replacement_func():
    return fused_softmax