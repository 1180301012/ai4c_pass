import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([1, 0, 2])
    tmp_4 = tmp_3.unsqueeze(0)
    return tmp_4


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": 64, "BLOCK_D": 16}, num_warps=1),
        triton.Config({"BLOCK_R": 128, "BLOCK_D": 16}, num_warps=1),
        triton.Config({"BLOCK_R": 256, "BLOCK_D": 16}, num_warps=2),
    ],
    key=["R", "D"],
)
@triton.jit
def _embedding_permute_unsqueeze_kernel(
    weight_ptr,
    index_ptr,
    out_ptr,
    R,
    W,
    D,
    weight_stride0,
    weight_stride1,
    index_stride0,
    index_stride1,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    d = tl.arange(0, BLOCK_D)

    r_mask = r < R
    d_mask = d < D

    i = r // W
    j = r % W

    idx = tl.load(index_ptr + i * index_stride0 + j * index_stride1, mask=r_mask, other=0)
    idx = idx.to(tl.int32)

    weight_offsets = idx[:, None] * weight_stride0 + d[None, :] * weight_stride1
    vals = tl.load(weight_ptr + weight_offsets, mask=r_mask[:, None] & d_mask[None, :], other=0)

    out_offsets = d[None, :] * R + r[:, None]
    tl.store(out_ptr + out_offsets, vals, mask=r_mask[:, None] & d_mask[None, :])


@torch.fx.wrap
def fused_embedding_permute_unsqueeze(in_0, in_1):
    weight = in_0
    index = in_1

    if not weight.is_cuda or weight.device != index.device:
        weight = weight.to(device=index.device)

    h = index.shape[0]
    w = index.shape[1]
    d = weight.shape[1]
    r = h * w

    out = torch.empty((1, d, h, w), device=index.device, dtype=weight.dtype)

    grid = lambda meta: (triton.cdiv(r, meta["BLOCK_R"]),)
    _embedding_permute_unsqueeze_kernel[grid](
        weight,
        index,
        out,
        r,
        w,
        d,
        weight.stride(0),
        weight.stride(1),
        index.stride(0),
        index.stride(1),
    )
    return out


def replacement_func():
    return fused_embedding_permute_unsqueeze