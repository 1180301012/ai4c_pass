import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches:
#   conv2d(x, w, b, (1,1), (0,0), (1,1), 1)
#   stack([conv], dim=0)
#   sum(dim=0)
#   cat([..., y], 1)
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.stack([conv2d], dim=0)
    tmp_4 = tmp_3.sum(dim=0)
    tmp_5 = torch.cat([tmp_4, in_3], 1)
    return tmp_5


# Normalize args to (bias, weight, conv_input, cat_input)
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=5, num_warps=8),
    ],
    key=["HW", "O", "K"],
)
@triton.jit
def _pointwise_conv_cat_head_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    HW,
    O,
    K,
    TOTAL_C,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_batch_base = pid_b * K * HW
    out_batch_base = pid_b * TOTAL_C * HW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_idx = k_iter + offs_k

        x_ptrs = x_ptr + x_batch_base + offs_m[:, None] + k_idx[None, :] * HW
        w_ptrs = w_ptr + k_idx[:, None] + offs_n[None, :] * K

        x_mask = (offs_m[:, None] < HW) & (k_idx[None, :] < K)
        w_mask = (k_idx[:, None] < K) & (offs_n[None, :] < O)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)

        k_iter += BLOCK_K

    b = tl.load(b_ptr + offs_n, mask=offs_n < O, other=0.0).to(tl.float32)
    acc += b[None, :]

    out_ptrs = out_ptr + out_batch_base + offs_m[:, None] + offs_n[None, :] * HW
    out_mask = (offs_m[:, None] < HW) & (offs_n[None, :] < O)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 2048}, num_warps=8),
        triton.Config({"BLOCK": 4096}, num_warps=8),
    ],
    key=["TAIL_ELEMS", "HEAD_C", "TOTAL_C"],
)
@triton.jit
def _copy_cat_tail_kernel(
    tail_ptr,
    out_ptr,
    TAIL_ELEMS,
    HEAD_C,
    TOTAL_C,
    HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TAIL_ELEMS

    tail_batch_base = pid_b * TAIL_ELEMS
    out_batch_base = pid_b * TOTAL_C * HW + HEAD_C * HW

    vals = tl.load(tail_ptr + tail_batch_base + offs, mask=mask, other=0.0)
    tl.store(out_ptr + out_batch_base + offs, vals, mask=mask)


@torch.fx.wrap
def fused_pointwise_conv_singleton_stack_sum_cat(bias, weight, conv_input, cat_input):
    bsz, in_c, h, w = conv_input.shape
    out_c = weight.shape[0]
    tail_c = cat_input.shape[1]
    hw = h * w

    out = torch.empty((bsz, out_c + tail_c, h, w), device=conv_input.device, dtype=conv_input.dtype)

    grid_tail = lambda META: (triton.cdiv(tail_c * hw, META["BLOCK"]), bsz)
    _copy_cat_tail_kernel[grid_tail](
        cat_input,
        out,
        tail_c * hw,
        out_c,
        out_c + tail_c,
        hw,
    )

    grid_head = lambda META: (triton.cdiv(hw, META["BLOCK_M"]), triton.cdiv(out_c, META["BLOCK_N"]), bsz)
    _pointwise_conv_cat_head_kernel[grid_head](
        conv_input,
        weight,
        bias,
        out,
        hw,
        out_c,
        in_c,
        out_c + tail_c,
    )

    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_pointwise_conv_singleton_stack_sum_cat