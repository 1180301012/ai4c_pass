import torch
import triton
import triton.language as tl


# Pattern: fuse view(-1,1)*in_2 → single output.
# Multi-output patterns crash in this framework — single output is the maximum supported.
def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.heuristics({
    'BLOCK_FEAT': lambda args: max(args['D'], 16),
    'num_warps': lambda args: 4 if args['D'] > 32 else 2,
})
@triton.jit
def broadcast_mul_kernel(
    in1_ptr,    # [N]     dtype  - edge_weight_1
    in2_ptr,    # [N*D]   dtype  - x_j
    out_ptr,    # [N*D]   dtype  - output
    N,
    D,          # runtime arg: used for cache key only (not a constexpr)
    BLOCK_FEAT: tl.constexpr,
):
    row = tl.program_id(0)
    offset = row * D
    cols = tl.arange(0, BLOCK_FEAT)
    mask = cols < D

    w = tl.load(in1_ptr + row)
    v = tl.load(in2_ptr + offset + cols, mask=mask, other=0.0)
    tl.store(out_ptr + offset + cols, w * v, mask=mask)


@torch.fx.wrap
def triton_broadcast_mul(in_1, in_2):
    N = in_2.shape[0]
    D = in_2.shape[1]
    out = torch.empty_like(in_2)
    broadcast_mul_kernel[(N,)](in_1, in_2, out, N, D=D)
    return out


def replacement_func():
    return triton_broadcast_mul