import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)


@triton.jit
def fused_linear_permute_kernel(
    in_3_ptr, weight_ptr, bias_ptr, out_ptr,
    N_spatial,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_spatial

    # Load C_in=3 input values for each spatial position
    x0 = tl.load(in_3_ptr + offsets * C_in, mask=mask, other=0.0)
    x1 = tl.load(in_3_ptr + offsets * C_in + 1, mask=mask, other=0.0)
    x2 = tl.load(in_3_ptr + offsets * C_in + 2, mask=mask, other=0.0)

    # For each output channel, compute dot product + bias and store
    for c in range(C_out):
        w0 = tl.load(weight_ptr + c * C_in)
        w1 = tl.load(weight_ptr + c * C_in + 1)
        w2 = tl.load(weight_ptr + c * C_in + 2)
        b = tl.load(bias_ptr + c)
        result = x0 * w0 + x1 * w1 + x2 * w2 + b
        tl.store(out_ptr + c * N_spatial + offsets, result, mask=mask)


@torch.fx.wrap
def fused_linear_permute(in_0, in_1, in_3):
    batch = in_3.shape[0]
    H = in_3.shape[1]
    W = in_3.shape[2]
    C_in = in_3.shape[3]
    C_out = in_1.shape[0]
    N_spatial = batch * H * W

    out = torch.empty((batch, C_out, H, W), dtype=in_3.dtype, device=in_3.device)

    BLOCK_SIZE = 256
    num_blocks = (N_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_linear_permute_kernel[(num_blocks,)](
        in_3, in_1, in_0, out,
        N_spatial, C_in, C_out, BLOCK_SIZE,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_linear_permute