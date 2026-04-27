import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match relu -> scale -> bias -> pad(right=1, bottom=1)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Fused Triton kernel
#
# Output layout : [N, C, H+1, W+1]  (same dtype as in_2)
#
# For output element (n, c, h, w):
#   h < H and w < W  ->  max(in2[n,c,h,w], 0) * scale + bias
#   otherwise        ->  0  (padding)
#
# Grid: (N*C*H_out, ceil(W_out / BLOCK_W))
# Each thread block processes one contiguous slice of BLOCK_W output columns.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 128}, num_warps=4),
        triton.Config({'BLOCK_W': 256}, num_warps=4),
        triton.Config({'BLOCK_W': 256}, num_warps=8),
        triton.Config({'BLOCK_W': 512}, num_warps=8),
        triton.Config({'BLOCK_W': 512}, num_warps=16),
    ],
    key=['H_out', 'W_out'],
)
@triton.jit
def _fused_relu_scale_bias_pad_kernel(
    in0_ptr,   # [1]  bias
    in1_ptr,   # [1]  scale
    in2_ptr,   # [N, C, H, W]  input
    out_ptr,   # [N, C, H+1, W+1]  output
    C, H, W,
    H_out, W_out,
    BLOCK_W: tl.constexpr,
):
    # pid_row identifies (n, c, h) together
    pid_row = tl.program_id(0)   # range: [0, N*C*H_out)
    pid_w   = tl.program_id(1)   # range: [0, ceil(W_out/BLOCK_W))

    # Decompose pid_row -> (n, c, h)
    h  = pid_row % H_out
    nc = pid_row // H_out
    c  = nc % C
    n  = nc // C

    # Column offsets for this block
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w    = w_offsets < W_out

    # Load scalar bias and scale (broadcast from shape [1])
    bias  = tl.load(in0_ptr).to(tl.float32)
    scale = tl.load(in1_ptr).to(tl.float32)

    # Valid = not in padding row, not in padding column
    is_valid = (h < H) & (w_offsets < W)

    # Load input values (guarded)
    in2_base   = ((n * C + c) * H + h) * W
    in2_val    = tl.load(
        in2_ptr + in2_base + w_offsets,
        mask=mask_w & is_valid,
        other=0.0,
    ).to(tl.float32)

    # relu + scale + bias; padding positions get 0
    relu_val = tl.maximum(in2_val, 0.0)
    result   = tl.where(is_valid & mask_w, relu_val * scale + bias, 0.0)

    # Store (Triton auto-converts float32 -> output dtype)
    out_base = ((n * C + c) * H_out + h) * W_out
    tl.store(out_ptr + out_base + w_offsets, result, mask=mask_w)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_relu_scale_bias_pad(in_0, in_1, in_2):
    N, C, H, W = in_2.shape
    H_out = H + 1
    W_out = W + 1

    out = torch.empty((N, C, H_out, W_out), dtype=in_2.dtype, device=in_2.device)

    # Grid: rows x column-blocks
    BLOCK_W_guess = 256
    grid_rows = N * C * H_out
    grid_w    = (W_out + BLOCK_W_guess - 1) // BLOCK_W_guess

    _fused_relu_scale_bias_pad_kernel[
        (grid_rows, grid_w)
    ](
        in_0, in_1, in_2, out,
        C, H, W,
        H_out, W_out,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-argument, returns the callable
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_relu_scale_bias_pad