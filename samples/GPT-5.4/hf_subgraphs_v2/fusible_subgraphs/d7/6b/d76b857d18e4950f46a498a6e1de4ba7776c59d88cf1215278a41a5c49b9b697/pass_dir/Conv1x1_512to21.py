import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


OUT_CHANNELS = 21
IN_CHANNELS = 512
HW_SIZE = 64 * 64


def pattern(in_0, in_1, in_2):
    return torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    ],
    key=["M"],
)
@triton.jit

def _conv1x1_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    HW: tl.constexpr,
    IC: tl.constexpr,
    OC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_O: tl.constexpr = 32,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    offs_hw = offs_m % HW
    offs_b = offs_m // HW

    offs_o = tl.arange(0, BLOCK_O)
    mask_o = offs_o < OC

    acc = tl.zeros((BLOCK_O, BLOCK_M), dtype=tl.float32)

    for k0 in range(0, IC, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < IC

        x_offsets = offs_b[None, :] * (IC * HW) + offs_k[:, None] * HW + offs_hw[None, :]
        w_offsets = offs_o[:, None] * IC + offs_k[None, :]

        x = tl.load(x_ptr + x_offsets, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
        w = tl.load(w_ptr + w_offsets, mask=mask_o[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(w, x)

    bias = tl.load(b_ptr + offs_o, mask=mask_o, other=0.0).to(tl.float32)
    acc += bias[:, None]

    out_offsets = offs_b[None, :] * (OC * HW) + offs_o[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=mask_o[:, None] & mask_m[None, :])


@torch.fx.wrap
def conv1x1_512to21_triton(bias, weight, feat):
    bias = unwrap_tensor(bias)
    weight = unwrap_tensor(weight)
    feat = unwrap_tensor(feat)

    bsz = feat.shape[0]
    out = torch.empty((bsz, OUT_CHANNELS, feat.shape[2], feat.shape[3]), device=feat.device, dtype=feat.dtype)
    m = bsz * feat.shape[2] * feat.shape[3]
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),)
    _conv1x1_kernel[grid](
        feat,
        weight,
        bias,
        out,
        m,
        HW=HW_SIZE,
        IC=IN_CHANNELS,
        OC=OUT_CHANNELS,
    )
    return out


def replacement_func():
    return conv1x1_512to21_triton