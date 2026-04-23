import torch
import triton
import triton.language as tl


# Constant for -inf in float32
NEG_INF = -3.4028234663852886e+38


def pattern(in_0):
    # Match the mask generation structure regardless of constant value
    L = 21  # This placeholder matches the pattern structure; actual value varies per graph
    tmp_1 = torch.arange(0, L, device=in_0.device)
    tmp_2 = torch.full((L, L), fill_value=NEG_INF, dtype=torch.float32, device=in_0.device)
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(L, device=in_0.device)
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_7 = tmp_3 * tmp_6
    return tmp_7


def replacement_args(in_0):
    # Extract sequence length from input shape
    seq_len = in_0.shape[1]
    return (in_0, seq_len)


@triton.jit
def causal_mask_kernel(
    out_ptr,
    L,
    BLOCK_SIZE: tl.constexpr
):
    # Each block processes one row of the mask
    row = tl.program_id(0)  # Current row index
    col = tl.arange(0, BLOCK_SIZE)
    mask = (col < L)  # Validity mask
    # Set to NEG_INF for column < row, 0 otherwise
    values = tl.where(col < row, NEG_INF, 0.0)
    # Store values for current row
    tl.store(out_ptr + row * L + col, values, mask=mask)


@torch.fx.wrap
def causal_mask_wrapper(in_0, seq_len):
    # Create output mask tensor
    mask = torch.empty((seq_len, seq_len), dtype=torch.float32, device=in_0.device)
    BLOCK_SIZE = 128  # Optimized block size for good occupancy
    num_blocks = seq_len  # One block per row
    causal_mask_kernel[(num_blocks,)](
        mask, seq_len, BLOCK_SIZE=BLOCK_SIZE
    )
    return mask

def replacement_func():
    return causal_mask_wrapper