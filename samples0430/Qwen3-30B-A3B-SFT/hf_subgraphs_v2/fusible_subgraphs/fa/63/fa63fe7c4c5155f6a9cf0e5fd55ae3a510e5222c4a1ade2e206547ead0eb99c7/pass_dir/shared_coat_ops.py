import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['TOTAL'],
)
@triton.jit
def coat_transpose_split_kernel(
    in_ptr, out0_ptr, out1_ptr, out2_ptr,
    N, S, C, W,
    SPLIT: tl.constexpr,
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: slice(in_2, 1:, dim=2) + transpose(-1,-2) + reshape + split

    Input  in_2: [1, 8, M, N] where M = S + 1 (S = H*W spatial positions, 1 CLS token)
    Output out0: [1, first,  H, W]
    Output out1: [1, second, H, W]
    Output out2: [1, third,  H, W]

    Mapping: in_2[0, h, n, s]  ->  out[0, h*SPLIT + (n-1), s//W, s%W]
    (after slicing, so n starts at 1 in in_2, giving n-1 in [0, S))
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < TOTAL

    # Decode output position (flat over C * H * W)
    hw = idx % (H * W)
    c  = (idx // (H * W)) % C

    # Map c -> (head, channel_within_head)
    head      = c // SPLIT
    chan_local = c % SPLIT

    # hw encodes spatial position:  hw = row * W + col
    row = hw // W
    col = hw % W

    # Corresponding position in in_2 (sequence index = chan_local + 1 since we skip CLS token)
    # in_2 shape [1, 8, M, N]; strides: [8*M*N, M*N, N, 1]
    in_idx = head * (S + 1) * N + (chan_local + 1) * N + row * W + col

    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)

    # Write to correct output tensor based on channel group
    is_g0 = c < SPLIT
    is_g1 = (c >= SPLIT) & (c < 2 * SPLIT)

    tl.store(out0_ptr + idx, val, mask=mask & is_g0)
    tl.store(out1_ptr + (idx - SPLIT * H * W), val, mask=mask & is_g1)
    tl.store(out2_ptr + (idx - 2 * SPLIT * H * W), val, mask=mask & ~is_g0 & ~is_g1)


@torch.fx.wrap
def dispatch_coat_fusion(in_0, in_1, in_2, route):
    # Decode route -> (SPLIT, W, H, C)
    if route == "38_7_7":
        SPLIT, W, H, C = 38, 7, 7, 152
    elif route == "54_7_7":
        SPLIT, W, H, C = 54, 7, 7, 216
    elif route == "64_7_7":
        SPLIT, W, H, C = 64, 7, 7, 256
    elif route == "80_7_7":
        SPLIT, W, H, C = 80, 7, 7, 320
    elif route == "38_14_14":
        SPLIT, W, H, C = 38, 14, 14, 152
    elif route == "54_14_14":
        SPLIT, W, H, C = 54, 14, 14, 216
    elif route == "64_14_14":
        SPLIT, W, H, C = 64, 14, 14, 256
    elif route == "38_28_28":
        SPLIT, W, H, C = 38, 28, 28, 152
    elif route == "54_28_28":
        SPLIT, W, H, C = 54, 28, 28, 216
    elif route == "128_12_12":
        SPLIT, W, H, C = 128, 12, 12, 512
    else:
        # Fallback: 38_7_7
        SPLIT, W, H, C = 38, 7, 7, 152

    N = in_2.shape[-1]
    S = H * W          # number of spatial positions after CLS token removal
    TOTAL = C * H * W  # elements per output tensor

    out0 = torch.empty(1, SPLIT, H, W, dtype=in_2.dtype, device=in_2.device)
    out1 = torch.empty(1, SPLIT, H, W, dtype=in_2.dtype, device=in_2.device)
    out2 = torch.empty(1, SPLIT, H, W, dtype=in_2.dtype, device=in_2.device)

    grid = lambda meta: ((TOTAL + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    coat_transpose_split_kernel[grid](
        in_2, out0, out1, out2,
        N, S, C, W,
        SPLIT=SPLIT,
        TOTAL=TOTAL,
    )

    return out0, out1, out2