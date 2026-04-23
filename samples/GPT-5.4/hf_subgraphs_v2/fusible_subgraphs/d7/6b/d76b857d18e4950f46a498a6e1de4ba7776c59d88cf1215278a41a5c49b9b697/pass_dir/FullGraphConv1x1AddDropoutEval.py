import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


OUT_CHANNELS = 21
IN_CHANNELS = 512
HW_SIZE = 64 * 64


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return (tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["B"],
)
@triton.jit
def _conv1x1_512_to_21_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B,
    BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_O: tl.constexpr = 32,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW_SIZE

    offs_o = tl.arange(0, BLOCK_O)
    mask_o = offs_o < OUT_CHANNELS

    x_batch_ptr = x_ptr + pid_b * IN_CHANNELS * HW_SIZE
    out_batch_ptr = out_ptr + pid_b * OUT_CHANNELS * HW_SIZE

    acc = tl.zeros((BLOCK_O, BLOCK_HW), dtype=tl.float32)

    for k0 in range(0, IN_CHANNELS, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < IN_CHANNELS

        x = tl.load(
            x_batch_ptr + offs_k[:, None] * HW_SIZE + offs_hw[None, :],
            mask=mask_k[:, None] & mask_hw[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_o[:, None] * IN_CHANNELS + offs_k[None, :],
            mask=mask_o[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc += tl.dot(w, x)

    bias = tl.load(b_ptr + offs_o, mask=mask_o, other=0.0).to(tl.float32)
    acc += bias[:, None]

    out = acc.to(tl.element_type(out_ptr))
    tl.store(
        out_batch_ptr + offs_o[:, None] * HW_SIZE + offs_hw[None, :],
        out,
        mask=mask_o[:, None] & mask_hw[None, :],
    )


@torch.fx.wrap
def full_graph_conv1x1_add_dropout_eval(bias, weight, feat, x_12, x_6):
    bias = unwrap_tensor(bias)
    weight = unwrap_tensor(weight)
    feat = unwrap_tensor(feat)
    x_12 = unwrap_tensor(x_12)
    x_6 = unwrap_tensor(x_6)

    add_out = torch.empty_like(x_6)
    conv_out = torch.empty((feat.shape[0], OUT_CHANNELS, feat.shape[2], feat.shape[3]), device=feat.device, dtype=feat.dtype)

    n_elements = add_out.numel()
    add_grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _add_kernel[add_grid](
        x_6,
        x_12,
        add_out,
        n_elements,
    )

    conv_grid = lambda meta: (feat.shape[0], triton.cdiv(HW_SIZE, meta["BLOCK_HW"]))
    _conv1x1_512_to_21_kernel[conv_grid](
        feat,
        weight,
        bias,
        conv_out,
        feat.shape[0],
    )

    return (add_out, conv_out)


def replacement_func():
    return full_graph_conv1x1_add_dropout_eval