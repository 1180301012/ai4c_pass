import torch
import triton
import triton.language as tl


@triton.jit
def _fused_max_softmax_kernel(
    input_ptr,
    output_ptr,
    total_rows,
    N,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """
    Fused numerically-stable softmax kernel.
    Each program handles one row of length N.
    Input/output dtype is controlled by DTYPE constexpr (tl.float16 / tl.bfloat16 / tl.float32).
    """
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load row data (float32 cast for numerical stability, matching PyTorch's internal behaviour)
    x = tl.load(input_ptr + row * N + col_offsets, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)

    # Numerically stable softmax: subtract row max before exp
    row_max = tl.max(x, axis=0)
    x = x - row_max
    x = tl.exp(x)
    # Zero out padding positions so they don't contribute to the sum
    x = tl.where(mask, x, 0.0)
    row_sum = tl.sum(x, axis=0)
    probs = x / row_sum

    # Store in the original dtype
    tl.store(output_ptr + row * N + col_offsets, probs.to(DTYPE), mask=mask)


@torch.fx.wrap
def fused_max_softmax_wrapper(in_0, in_1, batch, channel, dtype):
    """
    Fused kernel wrapper:
      1. Compute numerically stable softmax over last dim (fuses max + subtract + softmax)
      2. Perform view(B, C, -1) on in_1 (zero-copy reshape)
    """
    N = in_0.shape[-1]          # 512 for all graphs
    total_rows = batch * channel

    # Allocate output tensor (same dtype/device as input)
    out = torch.empty_like(in_0)

    # Launch fused kernel: one program per row
    _fused_max_softmax_kernel[(total_rows,)](
        in_0, out,
        total_rows, N,
        BLOCK_SIZE=N,
        DTYPE=dtype,
    )

    # Zero-copy view reshape (equivalent to in_1.view(batch, channel, -1))
    tmp_5 = in_1.view(batch, channel, -1)

    return out, tmp_5