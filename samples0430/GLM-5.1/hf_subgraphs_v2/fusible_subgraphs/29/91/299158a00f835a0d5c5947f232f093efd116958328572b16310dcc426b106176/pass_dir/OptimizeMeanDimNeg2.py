import torch
import triton
import triton.language as tl


def pattern(x):
    return x.mean(-2)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 16}, num_warps=4),
        triton.Config({'BLOCK_S': 32}, num_warps=4),
        triton.Config({'BLOCK_S': 64}, num_warps=4),
        triton.Config({'BLOCK_S': 16}, num_warps=8),
        triton.Config({'BLOCK_S': 32}, num_warps=8),
        triton.Config({'BLOCK_S': 64}, num_warps=8),
    ],
    key=['B', 'S', 'D'],
)
@triton.jit
def mean_neg2_kernel(
    x_ptr, out_ptr,
    B, S, D,
    stride_x0, stride_x1, stride_x2,
    stride_o0, stride_o1,
    BLOCK_S: tl.constexpr,
):
    # Per-element approach: each program computes one (b, d) output element
    # This provides B * D programs for good GPU utilization
    pid = tl.program_id(0)
    pid_b = pid // D
    pid_d = pid % D

    acc = 0.0

    for s_start in range(0, S, BLOCK_S):
        off_s = s_start + tl.arange(0, BLOCK_S)
        mask_s = off_s < S

        x_vals = tl.load(
            x_ptr + pid_b * stride_x0 + off_s * stride_x1 + pid_d * stride_x2,
            mask=mask_s, other=0.0
        ).to(tl.float32)

        acc += tl.sum(x_vals)

    result = acc / S

    tl.store(out_ptr + pid_b * stride_o0 + pid_d * stride_o1, result)


@torch.fx.wrap
def triton_mean_neg2(x):
    B, S, D = x.shape

    out = torch.empty((B, D), device=x.device, dtype=x.dtype)

    grid = lambda META: (B * D,)

    mean_neg2_kernel[grid](
        x_ptr=x, out_ptr=out,
        B=B, S=S, D=D,
        stride_x0=x.stride(0), stride_x1=x.stride(1), stride_x2=x.stride(2),
        stride_o0=out.stride(0), stride_o1=out.stride(1),
    )

    return out


def replacement_func():
    return triton_mean_neg2