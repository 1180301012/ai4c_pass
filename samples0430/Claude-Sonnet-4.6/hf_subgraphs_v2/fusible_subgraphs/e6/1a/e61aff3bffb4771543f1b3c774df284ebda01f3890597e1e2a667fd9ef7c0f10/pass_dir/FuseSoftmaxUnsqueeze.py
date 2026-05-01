import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row. Each program computes softmax over one row of length n_cols.
    BLOCK_SIZE must be a power of 2 >= n_cols.
    """
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load the row; pad out-of-bounds with -inf so they don't affect max/sum
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=float('-inf'))

    # Accumulate in float32 for numerical stability (important for fp16/bf16)
    row_f32 = row.to(tl.float32)

    # Numerically-stable softmax: subtract row max before exp
    row_max = tl.max(row_f32, axis=0)
    row_shifted = row_f32 - row_max
    exp_row = tl.exp(row_shifted)
    sum_exp = tl.sum(exp_row, axis=0)
    softmax_row = exp_row / sum_exp

    # Cast back to original dtype
    softmax_row = softmax_row.to(row.dtype)

    # Store (only valid positions)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_row, mask=mask)


# ── Pattern to match ─────────────────────────────────────────────────────────

def pattern(x):
    """
    Matches:
        t    = torch.nn.functional.softmax(x, 2, _stacklevel=5)
        out  = t.unsqueeze(-1)
    Returns the final unsqueezed tensor.
    """
    t = torch.nn.functional.softmax(x, 2, _stacklevel=5)
    return t.unsqueeze(-1)


def replacement_args(x):
    return (x,)


# ── Replacement ───────────────────────────────────────────────────────────────

@torch.fx.wrap
def triton_softmax_unsqueeze(x):
    """
    Fused softmax(dim=2) + unsqueeze(-1) implemented in Triton.

    x is expected to be a 3-D tensor of shape [N, 1, K] (or any shape whose
    last dimension is the softmax dimension).
    Output shape: x.shape + (1,)  i.e. [N, 1, K, 1].
    """
    orig_shape = x.shape
    n_cols = int(orig_shape[-1])
    n_rows = int(x.numel()) // n_cols

    # Compute BLOCK_SIZE: smallest power of 2 that is >= n_cols
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2
    # Hard upper-bound so the kernel is always compilable on current hardware
    BLOCK_SIZE = min(BLOCK_SIZE, 65536)

    # num_warps: aim for at most 1024 threads (32 warps) per program
    num_warps = min(max(BLOCK_SIZE // 32, 1), 32)

    # Reshape to 2-D view [n_rows, n_cols] for the kernel
    x_2d = x.reshape(n_rows, n_cols)
    out_2d = torch.empty_like(x_2d)

    _softmax_kernel[(n_rows,)](
        x_2d, out_2d,
        x_2d.stride(0), out_2d.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # unsqueeze(-1) is a zero-cost view: [n_rows, n_cols] → orig_shape + (1,)
    return out_2d.reshape(orig_shape).unsqueeze(-1)


def replacement_func():
    return triton_softmax_unsqueeze