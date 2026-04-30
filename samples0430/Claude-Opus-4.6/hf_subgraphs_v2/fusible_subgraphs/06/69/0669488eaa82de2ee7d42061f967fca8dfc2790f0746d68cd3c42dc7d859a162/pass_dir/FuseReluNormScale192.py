import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim=-1, keepdim=True)
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min=1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return (tmp_7,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
    ],
    key=['num_rows'],
)
@triton.jit
def fused_relu_normalize_kernel_192(
    in_ptr,
    g_ptr,
    out_ptr,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    SCALE: tl.constexpr = 0.07216878364870322
    D: tl.constexpr = 192

    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    # Load row
    row_start = row_idx * D
    x = tl.load(in_ptr + row_start + col_offsets, mask=mask, other=0.0)

    # Cast to float32 for computation
    x_f32 = x.to(tl.float32)

    # ReLU
    x_f32 = tl.maximum(x_f32, 0.0)

    # L2 norm
    norm_sq = tl.sum(x_f32 * x_f32, axis=0)
    norm = tl.sqrt(norm_sq)

    # Scale and clamp
    scaled_norm = norm * SCALE
    clamped_norm = tl.maximum(scaled_norm, 1e-5)

    # Load scalar g and compute result
    g = tl.load(g_ptr).to(tl.float32)
    result_f32 = x_f32 * (g / clamped_norm)

    # Cast back and store
    result = result_f32.to(x.dtype)
    tl.store(out_ptr + row_start + col_offsets, result, mask=mask)


@torch.fx.wrap
def fused_relu_normalize_192(in_0, in_1):
    B = in_1.shape[0]
    C = in_1.shape[1]
    D = 192  # in_1.shape[2] * in_1.shape[3] = 16 * 12

    num_rows = B * C
    out = torch.empty(B, C, D, dtype=in_1.dtype, device=in_1.device)

    grid = (num_rows,)
    fused_relu_normalize_kernel_192[grid](
        in_1, in_0, out,
        num_rows,
    )

    return (out,)


def replacement_func():
    return fused_relu_normalize_192