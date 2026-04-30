import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_P": 128, "BLOCK_C": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_P": 256, "BLOCK_C": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_P": 256, "BLOCK_C": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_P": 512, "BLOCK_C": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_P": 512, "BLOCK_C": 64}, num_warps=8, num_stages=3),
    ],
    key=["B"],
)
@triton.jit
def _conv1x1_reduce_kernel(
    x_ptr,
    w4_ptr,
    bias_ptr,
    score_ptr,
    stride_xb,
    stride_xc,
    stride_sb,
    B,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    if pid_b >= B:
        return

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    acc = tl.zeros([BLOCK_P], dtype=tl.float32)
    bias = tl.load(bias_ptr).to(tl.float32)

    for c0 in range(0, 512, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        x_ptrs = x_ptr + pid_b * stride_xb + offs_c[:, None] * stride_xc + offs_p[None, :]
        x = tl.load(x_ptrs).to(tl.float32)
        w = tl.load(w4_ptr + offs_c).to(tl.float32)
        acc += tl.sum(x * w[:, None], axis=0)

    tl.store(score_ptr + pid_b * stride_sb + offs_p, acc + bias)


@triton.jit
def _softmax_4096_kernel(
    score_ptr,
    out_ptr,
    stride_sb,
    stride_ob,
    B,
):
    pid_b = tl.program_id(0)
    if pid_b >= B:
        return

    offs = tl.arange(0, 4096)
    row = tl.load(score_ptr + pid_b * stride_sb + offs)
    row = row - tl.max(row, axis=0)
    numer = tl.exp(row)
    denom = tl.sum(numer, axis=0)
    tl.store(out_ptr + pid_b * stride_ob + offs, numer / denom)


@torch.fx.wrap
def fused_conv1x1_spatial_softmax(bias, weight, x):
    batch = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]

    assert channels == 512
    assert height == 64
    assert width == 64
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert bias.numel() == 1
    assert weight.shape[0] == 1 and weight.shape[1] == 512 and weight.shape[2] == 1 and weight.shape[3] == 1

    scores = torch.empty((batch, 4096), device=x.device, dtype=torch.float32)
    out = torch.empty((batch, 1, 4096), device=x.device, dtype=x.dtype)

    grid = lambda meta: (batch, 4096 // meta["BLOCK_P"])
    _conv1x1_reduce_kernel[grid](
        x,
        weight,
        bias,
        scores,
        x.stride(0),
        x.stride(1),
        scores.stride(0),
        batch,
    )
    _softmax_4096_kernel[(batch,)](
        scores,
        out,
        scores.stride(0),
        out.stride(0),
        batch,
        num_warps=8,
        num_stages=4,
    )
    return out