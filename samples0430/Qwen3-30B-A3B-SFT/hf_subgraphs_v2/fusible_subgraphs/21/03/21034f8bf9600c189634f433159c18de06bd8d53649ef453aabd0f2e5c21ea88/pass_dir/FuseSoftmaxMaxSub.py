import torch
import triton
import triton.language as tl


@triton.jit
def fused_max_softmax_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused numerically-stable softmax kernel.
    Each program handles one row of length N.
    Steps:
      1. Load row of x
      2. Find max(x)  (for numerical stability)
      3. Compute x - max(x)
      4. Compute exp(x - max(x))
      5. Compute sum(exp(...))
      6. Divide to get softmax values
      7. Store results in the original dtype
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load raw input values for this row
    x_raw = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float('inf'))

    # Upcast to float32 for accumulation precision (important for fp16/bf16)
    x = x_raw.to(tl.float32)

    # Numerically stable max
    x_max = tl.max(x, axis=0)

    # Shift: x - max(x)
    x_shifted = x - x_max

    # Exponentiate
    x_exp = tl.exp(x_shifted)

    # Normalise: sum of exp
    x_sum = tl.sum(x_exp, axis=0)

    # Softmax values
    out = x_exp / x_sum

    # Store back in the original dtype
    tl.store(out_ptr + row_start + offsets, out.to(x_raw.dtype), mask=mask)


@torch.fx.wrap
def fused_max_softmax(x):
    """
    Fused: max(x, dim=-1, keepdim=True)[0] expanded and subtracted,
    then softmax along last dimension.
    Input shape:  [*, N]  (any number of leading dims, N is the last dim)
    Output shape: same as input
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    num_rows = x.numel() // N

    out = torch.empty_like(x)

    # BLOCK_SIZE must be a power-of-2 >= N; for N=512 this is 512
    BLOCK_SIZE = triton.next_power_of_2(N)

    fused_max_softmax_kernel[(num_rows,)](
        x_ptr=x,
        out_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern, replacement_args, replacement_func
# ---------------------------------------------------------------------------

def pattern(in_0):
    """
    Matches the numerically-stable softmax computation:
      max → getitem(0) → expand_as → subtract → softmax
    This pattern covers ALL batch sizes and all supported dtypes
    (float32, float16, bfloat16) because it uses no shape-specific constants.
    """
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_max_softmax