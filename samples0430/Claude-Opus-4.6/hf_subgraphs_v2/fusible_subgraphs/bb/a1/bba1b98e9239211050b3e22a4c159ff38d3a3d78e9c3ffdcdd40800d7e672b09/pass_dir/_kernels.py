import torch
import triton
import triton.language as tl


@triton.jit
def _conv1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C, N,
    stride_b, stride_c,
    TILE_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)

    n_start = tile_idx * TILE_N
    n_offsets = n_start + tl.arange(0, TILE_N)

    acc = tl.zeros([TILE_N], dtype=tl.float32)
    base = batch_idx * stride_b

    for c in range(C):
        w = tl.load(weight_ptr + c).to(tl.float32)
        x = tl.load(input_ptr + base + c * stride_c + n_offsets).to(tl.float32)
        acc += x * w

    bias_val = tl.load(bias_ptr).to(tl.float32)
    acc += bias_val

    tl.store(output_ptr + batch_idx * N + n_offsets, acc)


@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N

    x = tl.load(input_ptr + row_idx * N + offsets, mask=mask, other=float('-inf')).to(tl.float32)

    max_val = tl.max(x, axis=0)
    x = x - max_val
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    result = exp_x / sum_exp

    tl.store(output_ptr + row_idx * N + offsets, result, mask=mask)


@torch.fx.wrap
def fused_dispatch(x, weight, bias, route):
    B = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    N = H * W

    stride_b = C * N
    stride_c = N

    # Step 1: Conv 1x1
    TILE_N = 256
    num_tiles = N // TILE_N

    intermediate = torch.empty((B, N), dtype=torch.float32, device=x.device)

    _conv1x1_kernel[(B, num_tiles)](
        x, weight, bias, intermediate,
        B, C, N,
        stride_b, stride_c,
        TILE_N=TILE_N,
        num_warps=4,
        num_stages=2,
    )

    # Step 2: Softmax
    output = torch.empty((B, 1, N), dtype=x.dtype, device=x.device)
    BLOCK_N = triton.next_power_of_2(N)

    _softmax_kernel[(B,)](
        intermediate, output,
        N,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return output