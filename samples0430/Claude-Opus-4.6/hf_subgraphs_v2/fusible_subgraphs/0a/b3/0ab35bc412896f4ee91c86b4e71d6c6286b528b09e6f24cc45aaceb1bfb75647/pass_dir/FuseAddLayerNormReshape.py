import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)
    tmp_4 = tmp_3.reshape(1, 2, 2, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.permute(0, 2, 3, 1)
    tmp_8 = tmp_7.reshape(1, -1, 128)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_add_layernorm_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr, out_ptr,
    N: tl.constexpr,
    NUM_ROWS: tl.constexpr,
):
    # Single block processes all rows using 2D tiles
    row_offsets = tl.arange(0, NUM_ROWS)[:, None]  # [4, 1]
    col_offsets = tl.arange(0, N)[None, :]  # [1, 128]
    offsets = row_offsets * N + col_offsets  # [4, 128]

    # Load and add
    x = tl.load(in_3_ptr + offsets).to(tl.float32)
    y = tl.load(in_2_ptr + offsets).to(tl.float32)
    z = x + y

    # Layer norm per row
    mean = tl.sum(z, axis=1)[:, None] * (1.0 / N)
    diff = z - mean
    var = tl.sum(diff * diff, axis=1)[:, None] * (1.0 / N)
    inv_std = tl.rsqrt(var + 1e-5)
    normalized = diff * inv_std

    # Scale and shift (broadcast weight/bias across rows)
    weight = tl.load(weight_ptr + col_offsets).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets).to(tl.float32)
    out = normalized * weight + bias

    tl.store(out_ptr + offsets, out)


@torch.fx.wrap
def fused_add_layernorm(in_0, in_1, in_2, in_3):
    out = torch.empty_like(in_2)
    fused_add_layernorm_kernel[(1,)](
        in_2, in_3, in_1, in_0, out,
        N=128,
        NUM_ROWS=4,
        num_warps=4,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_add_layernorm