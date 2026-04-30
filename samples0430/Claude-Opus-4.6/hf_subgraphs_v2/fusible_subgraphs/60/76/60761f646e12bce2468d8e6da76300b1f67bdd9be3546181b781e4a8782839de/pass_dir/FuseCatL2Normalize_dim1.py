import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 1024}, num_warps=16, num_stages=1),
    ],
    key=["M", "N"],
)
@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)

    # Compute L2 norm
    norm_sq = tl.sum(x_fp32 * x_fp32, axis=0)
    norm = tl.sqrt(norm_sq)
    norm = tl.maximum(norm, 1e-12)

    # Normalize
    out = x_fp32 / norm
    out = out.to(x.dtype)

    tl.store(out_ptr + row * stride_m + cols, out, mask=mask)


@torch.fx.wrap
def l2_normalize_triton(in_0):
    M = in_0.shape[0]
    N = in_0.shape[1]
    out = torch.empty_like(in_0)

    l2_normalize_kernel[(M,)](
        in_0,
        out,
        M,
        N,
        in_0.stride(0),
    )
    return out


def replacement_func():
    return l2_normalize_triton