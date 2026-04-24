import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: flatten(2) + transpose(1,2) — right after conv3d output.
# This confirms the framework matches and gives a simple Triton-based pass.
# ---------------------------------------------------------------------------

def pattern(x):
    tmp_7 = x.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    return tmp_8


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: fused flatten + transpose
#   Input:  [B, C, T, H, W]  (conv3d output)
#   Output: [B, T*H*W, C]    (contiguous after transpose)
# For our model: [1, 768, 10, 224, 224] -> [1, 10*224*224, 768]
# We'll handle a slice since T*H*W = 501760 >> BLOCK_SIZE, but here N_COLS=C
# and we need to transpose row-major to col-major.
# This is essentially a 2D transpose: write output[b, s, c] = input[b, c, s]
# For simplicity we transpose one batch row at a time.
# ---------------------------------------------------------------------------

@triton.jit
def _flatten_transpose_kernel(
    x_ptr, out_ptr,
    N_rows,           # T * H * W  (sequence length)
    N_COLS,           # C          (channel / embedding dim)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (seq_pos, channel_chunk)
    # Grid is 2D: (N_rows, ceil(N_COLS / BLOCK_SIZE))
    pid_row  = tl.program_id(0)   # seq position
    pid_col  = tl.program_id(1)   # channel block
    col_start = pid_col * BLOCK_SIZE
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    # Input  layout: [C, N_rows]  (B=1 assumed)
    #   x_ptr[c, s] = x_ptr + c * N_rows + s
    # Output layout: [N_rows, C]  (B=1 assumed)
    #   out_ptr[s, c] = out_ptr + s * N_COLS + c
    # We read x[cols, pid_row] and write out[pid_row, cols]
    x = tl.load(x_ptr + cols * N_rows + pid_row, mask=mask, other=0.0)
    tl.store(out_ptr + pid_row * N_COLS + cols, x, mask=mask)


@torch.fx.wrap
def fused_flatten_transpose(x):
    # x: [B, C, *spatial]
    # flatten(2) -> [B, C, S]  where S = product of all spatial dims >= 3
    # transpose(1,2) -> [B, S, C]
    B  = x.shape[0]
    C  = x.shape[1]
    S  = x.numel() // (B * C)   # product of dims 2+

    out = torch.empty((B, S, C), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 64
    grid = (S, (C + BLOCK_SIZE - 1) // BLOCK_SIZE)

    _flatten_transpose_kernel[grid](
        x, out,
        S, C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


def replacement_func():
    return fused_flatten_transpose