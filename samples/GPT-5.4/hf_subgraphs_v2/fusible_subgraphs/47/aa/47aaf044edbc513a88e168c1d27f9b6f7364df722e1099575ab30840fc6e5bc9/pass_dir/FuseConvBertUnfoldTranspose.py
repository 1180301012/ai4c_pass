import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 8}, num_warps=2),
        triton.Config({"BLOCK_C": 16}, num_warps=2),
        triton.Config({"BLOCK_C": 32}, num_warps=4),
        triton.Config({"BLOCK_C": 64}, num_warps=4),
    ],
    key=["C", "L"],
)
@triton.jit
def _convbert_unfold_transpose_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    L,
    stride_xn,
    stride_xc,
    stride_xl,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_nl = tl.program_id(1)

    n = pid_nl // L
    l = pid_nl % L

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_k = tl.arange(0, 9)
    pos_l = l + offs_k - 4

    mask_c = offs_c < C
    mask_l = (pos_l >= 0) & (pos_l < L)
    mask = mask_c[:, None] & mask_l[None, :] & (n < N)

    x_offsets = n * stride_xn + offs_c[:, None] * stride_xc + pos_l[None, :] * stride_xl
    vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

    out_row_base = (n * L + l) * (C * 9)
    out_offsets = out_row_base + offs_c[:, None] * 9 + offs_k[None, :]
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@torch.fx.wrap
def fused_convbert_unfold_transpose(in_0):
    shape = in_0.shape
    N = shape[0]
    C = shape[1]
    L = shape[2]

    out = torch.empty((N, L, C * 9), device=in_0.device, dtype=in_0.dtype)

    stride_x = in_0.stride()
    stride_xn = stride_x[0]
    stride_xc = stride_x[1]
    stride_xl = stride_x[2]

    grid = lambda meta: (triton.cdiv(C, meta["BLOCK_C"]), N * L)
    _convbert_unfold_transpose_kernel[grid](
        in_0,
        out,
        N,
        C,
        L,
        stride_xn,
        stride_xc,
        stride_xl,
    )
    return out


def replacement_func():
    return fused_convbert_unfold_transpose