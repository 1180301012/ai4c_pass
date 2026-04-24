import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    conv2d = torch.ops.aten.convolution.default(in_2, in_1, in_0, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    tmp_3 = torch.ops.aten.view.default(conv2d, [4, 1, 192])
    tmp_4 = torch.ops.aten._softmax.default(tmp_3, 2, False)
    tmp_5 = torch.ops.aten.unsqueeze.default(tmp_4, -1)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64},   num_warps=1),
        triton.Config({'BLOCK_N': 128},  num_warps=2),
        triton.Config({'BLOCK_N': 256},  num_warps=4),
        triton.Config({'BLOCK_N': 512},  num_warps=4),
        triton.Config({'BLOCK_N': 512},  num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=4),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_warps=8),
        triton.Config({'BLOCK_N': 4096}, num_warps=8),
        triton.Config({'BLOCK_N': 4096}, num_warps=16),
    ],
    key=['N', 'C'],
)
@triton.jit
def fused_conv1x1_softmax_kernel(
    input_ptr,   # [B, C, H*W]
    weight_ptr,  # [C]
    bias_ptr,    # [1]
    output_ptr,  # [B, 1, N, 1]
    B, C, N,
    BLOCK_N: tl.constexpr,
):
    b     = tl.program_id(0)
    blk   = tl.program_id(1)
    n_start = blk * BLOCK_N
    offsets = n_start + tl.arange(0, BLOCK_N)
    mask    = offsets < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for c_id in range(C):
        w = tl.load(weight_ptr + c_id).to(tl.float32)
        x = tl.load(input_ptr + b * C * N + c_id * N + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = acc + w * x

    bias_val = tl.load(bias_ptr).to(tl.float32)
    acc = acc + bias_val

    neg_inf  = tl.full([BLOCK_N], float('-inf'), dtype=tl.float32)
    x_safe   = tl.where(mask, acc, neg_inf)
    max_val  = tl.max(x_safe, axis=0)
    exp_x    = tl.exp(x_safe - max_val)
    exp_x    = tl.where(mask, exp_x, 0.0)
    sum_exp  = tl.sum(exp_x, axis=0)
    result   = exp_x / sum_exp

    tl.store(output_ptr + b * N + offsets, result, mask=mask)


@torch.fx.wrap
def fused_softmax_unsqueeze(in_0, in_1, in_2):
    """Fused conv1x1 + view + softmax + unsqueeze → [B, 1, N, 1]."""
    B = in_2.shape[0]; C = in_2.shape[1]
    H = in_2.shape[2]; W = in_2.shape[3]; N = H * W
    out = torch.empty((B, 1, N, 1), dtype=in_2.dtype, device=in_2.device)
    grid = lambda meta: (B, triton.cdiv(N, meta['BLOCK_N']))
    fused_conv1x1_softmax_kernel[grid](in_2, in_1, in_0, out, B, C, N)
    return out


def replacement_func():
    return fused_softmax_unsqueeze