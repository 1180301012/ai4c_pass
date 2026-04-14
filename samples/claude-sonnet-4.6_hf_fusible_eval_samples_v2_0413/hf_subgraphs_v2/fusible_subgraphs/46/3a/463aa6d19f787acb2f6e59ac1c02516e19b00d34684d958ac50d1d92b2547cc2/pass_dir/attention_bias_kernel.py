import triton
import triton.language as tl


@triton.jit
def causal_attention_bias_kernel(
    in_0_ptr,   # int64 ptr to [1, N] attention mask
    out_ptr,    # float32 ptr to [1, 1, N, N] output
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Each program (indexed by pid_i) handles one row i of the NxN output.
    Computes the combined causal + padding attention bias mask:
      val[i,j] = NEG_INF if (j > i) or (in_0[j] == 0), else 0.0
    Rows where all positions [0..i] are padding are zeroed out entirely.
    """
    NEG_INF = -3.4028234663852886e+38

    pid_i = tl.program_id(0)         # row index

    j = tl.arange(0, BLOCK_N)        # column indices [0..BLOCK_N)
    mask_j = j < N                    # valid column mask

    # Load the attention mask for columns j; default to 1 (valid) for OOB
    in_0_vals = tl.load(in_0_ptr + j, mask=mask_j, other=1)   # int64

    # ---- Compute causal mask value ----------------------------------------
    # -NEG_INF where j > pid_i (future positions), else 0.0
    causal = tl.where(j > pid_i, NEG_INF, 0.0).to(tl.float32)

    # ---- Apply padding mask --------------------------------------------------
    # For present/past positions (j <= pid_i), mask out padding tokens
    val = tl.where(
        (j <= pid_i) & (in_0_vals == 0) & mask_j,
        NEG_INF,
        causal,
    ).to(tl.float32)

    # ---- Row validity -------------------------------------------------------
    # Row i is "valid" if at least one position in [0..pid_i] is non-padding
    valid_in_past = tl.sum(
        tl.where((j <= pid_i) & mask_j & (in_0_vals != 0), 1, 0),
        axis=0,
    )
    row_valid = valid_in_past > 0

    # If the entire row is masked (all-NEG_INF), zero it out
    out_val = tl.where(row_valid, val, tl.zeros_like(val))

    # ---- Store ---------------------------------------------------------------
    # Output layout: [1, 1, N, N] → row pid_i starts at offset pid_i * N
    tl.store(out_ptr + pid_i * N + j, out_val, mask=mask_j)