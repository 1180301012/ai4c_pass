import torch
import triton
import triton.language as tl


# Layer norm kernel with autotune for optimal block sizes per N
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32}, num_warps=2),
        triton.Config({'BLOCK_N': 64}, num_warps=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    eps, total_rows, N,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= total_rows:
        return

    row_offset = row_idx * N

    # Pass 1: compute statistics (single read of x)
    sum_x = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum_x2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        offsets = off + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(X_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x += x
        sum_x2 += x * x
    mean = tl.sum(sum_x) / N
    var = tl.sum(sum_x2) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize (re-read x, load w/b, store y)
    for off in range(0, N, BLOCK_N):
        offsets = off + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(X_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y_ptr + row_offset + offsets, y, mask=mask)


@torch.fx.wrap
def triton_layer_norm(x, weight, bias, N, eps=1e-12):
    total_rows = x.numel() // N
    y = torch.empty_like(x)
    grid = (total_rows,)
    layer_norm_kernel[grid](
        x, weight, bias, y,
        eps, total_rows, N,
    )
    return y


@torch.fx.wrap
def dispatch_wrapper(in_0, in_1, in_4, route):
    if route == "route_32":
        return triton_layer_norm(in_4, in_1, in_0, 32)
    elif route == "route_384":
        return triton_layer_norm(in_4, in_1, in_0, 384)
    elif route == "route_768":
        return triton_layer_norm(in_4, in_1, in_0, 768)
    else:
        raise ValueError(f"Unknown route: {route}")