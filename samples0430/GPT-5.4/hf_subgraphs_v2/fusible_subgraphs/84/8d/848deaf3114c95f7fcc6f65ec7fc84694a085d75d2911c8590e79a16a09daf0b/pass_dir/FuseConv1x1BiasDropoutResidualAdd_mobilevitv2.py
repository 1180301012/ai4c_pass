import torch
import triton
import triton.language as tl


# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.dropout(conv2d, 0.0, False, False)
    tmp_4 = tmp_3 + in_2
    return tmp_4


# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
    ],
    key=[],
)
@triton.jit
def _fused_conv1x1_bias_residual_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    residual_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = 1024 // BLOCK_M
    num_pid_n = 128 // BLOCK_N
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, 256, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        w_tile = tl.load(w_ptr + offs_n[:, None] * 256 + offs_k[None, :])
        x_tile = tl.load(x_ptr + offs_k[:, None] * 1024 + offs_m[None, :])
        acc += tl.dot(w_tile, x_tile)

    bias = tl.load(b_ptr + offs_n).to(tl.float32)
    acc += bias[:, None]

    residual = tl.load(residual_ptr + offs_n[:, None] * 1024 + offs_m[None, :])
    out = acc.to(residual.dtype) + residual
    tl.store(out_ptr + offs_n[:, None] * 1024 + offs_m[None, :], out)


@torch.fx.wrap
def triton_fused_conv1x1_bias_residual_add(in_0, in_1, in_2, in_3):
    # in_0: bias      [128]
    # in_1: weight    [128, 256, 1, 1]
    # in_2: residual  [1, 128, 4, 256]
    # in_3: input     [1, 256, 4, 256]
    out = torch.empty_like(in_2)

    grid = lambda meta: ((1024 // meta["BLOCK_M"]) * (128 // meta["BLOCK_N"]),)

    _fused_conv1x1_bias_residual_kernel[grid](
        in_3,
        in_1,
        in_0,
        in_2,
        out,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_fused_conv1x1_bias_residual_add