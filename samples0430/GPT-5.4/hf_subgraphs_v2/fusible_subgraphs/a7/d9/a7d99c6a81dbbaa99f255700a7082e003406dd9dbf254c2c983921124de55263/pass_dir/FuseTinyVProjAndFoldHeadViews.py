import torch
import triton
import triton.language as tl

HEAD_DIM = 64
OUT_DIM = 512
NUM_HEADS = OUT_DIM // HEAD_DIM


# Pattern matching function
# NOTE: Must mirror model.py exactly for the tmp_10-producing subgraph.
def pattern(in_0: torch.Tensor, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_10 = tmp_6.contiguous()
    return tmp_10


def replacement_args(in_0: torch.Tensor, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.jit
def tiny_vproj_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in tl.static_range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    out = acc + bias
    tl.store(out_ptr + offs_n, out, mask=mask_n)


@torch.fx.wrap
def fused_tiny_vproj_contiguous(in_0, in_1, in_3):
    out = torch.empty((1, NUM_HEADS, 1, HEAD_DIM), dtype=in_3.dtype, device=in_3.device)

    tiny_vproj_kernel[(8,)](
        in_3,
        in_1,
        in_0,
        out,
        N=OUT_DIM,
        K=OUT_DIM,
        BLOCK_N=64,
        BLOCK_K=64,
    )

    return out


def replacement_func():
    return fused_tiny_vproj_contiguous