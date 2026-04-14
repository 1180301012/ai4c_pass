import triton
import triton.language as tl


@triton.jit
def build_causal_mask_kernel(
    out_ptr,
    N,                      # sequence length (runtime)
    BLOCK_N: tl.constexpr,  # power-of-2 >= N
):
    """
    One program per row i. Writes NEG_INF where j > i, 0 elsewhere.
    Output shape: [N, N] float32.
    """
    NEG_INF = -3.4028234663852886e+38
    pid_i = tl.program_id(0)

    j      = tl.arange(0, BLOCK_N)
    mask_j = j < N

    val = tl.where(j > pid_i, NEG_INF, 0.0).to(tl.float32)
    tl.store(out_ptr + pid_i * N + j, val, mask=mask_j)