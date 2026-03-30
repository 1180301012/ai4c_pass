import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Pattern: relu -> flatten(_, 2) -> norm(dim=-1,keepdim=True)
#          -> * 0.07216878364870322 -> clamp(min=1e-5) -> / -> * in_0
# -----------------------------------------------------------------------
def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# -----------------------------------------------------------------------
# Fused Triton kernel
#   Grid: (rows,)  where  rows = B * C
#   Each program handles one row of the [rows, N] logical 2-D layout.
# -----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _fused_relu_norm_kernel_0p07216(
    in0_ptr,   # [1]  learnable scale weight (same dtype as in_1)
    in1_ptr,   # [rows, N]  raw input (before relu/flatten)
    out_ptr,   # [rows, N]  output
    N,         # inner dimension = H * W
    BLOCK_SIZE: tl.constexpr,
):
    SCALE = 0.07216878364870322

    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    row_start = row_idx * N

    # Load row (masked to avoid OOB); cast to fp32 for stable arithmetic
    x = tl.load(in1_ptr + row_start + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # ReLU
    x_f32 = tl.maximum(x_f32, 0.0)

    # L2 norm along the row
    norm_val = tl.sqrt(tl.sum(x_f32 * x_f32, axis=0))

    # Scale and clamp
    norm_clamped = tl.maximum(norm_val * SCALE, 1e-5)

    # Learnable weight (scalar, shape [1])
    w = tl.load(in0_ptr).to(tl.float32)

    # Normalise and apply weight
    out_f32 = (x_f32 / norm_clamped) * w

    # Store result in original dtype
    tl.store(out_ptr + row_start + offsets, out_f32.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_relu_norm_0p07216(in_0, in_1):
    """Replacement callable for the 0.07216... pattern."""
    # in_1 shape: [B, C, H, W]  (4-D)
    B, C = in_1.shape[0], in_1.shape[1]
    rows = B * C
    N = in_1.numel() // rows          # H * W

    in1_cont = in_1.contiguous()
    out = torch.empty((rows, N), dtype=in_1.dtype, device=in_1.device)

    _fused_relu_norm_kernel_0p07216[(rows,)](
        in_0,
        in1_cont,
        out,
        N,
    )

    return out.view(B, C, N)


def replacement_func():
    return fused_relu_norm_0p07216