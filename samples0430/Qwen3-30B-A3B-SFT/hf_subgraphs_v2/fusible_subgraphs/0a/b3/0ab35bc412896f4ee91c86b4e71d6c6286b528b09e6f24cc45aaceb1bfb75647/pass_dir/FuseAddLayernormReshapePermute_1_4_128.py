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


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=1),
    ],
    key=['N'],
)
@triton.jit
def fused_add_layernorm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: tl.constexpr,
    eps,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, N)

    # Load inputs and compute element-wise add
    x = tl.load(x_ptr + row_idx * N + cols).to(tl.float32)
    y = tl.load(y_ptr + row_idx * N + cols).to(tl.float32)
    z = x + y

    # Compute layer norm: mean and variance over the row
    mean = tl.sum(z, axis=0) / N
    diff = z - mean
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = diff * inv_std

    # Apply weight and bias
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    bias = tl.load(bias_ptr + cols).to(tl.float32)
    out = normalized * weight + bias

    # Store result (Triton auto-converts float32 → output dtype)
    tl.store(out_ptr + row_idx * N + cols, out)


@torch.fx.wrap
def fused_add_layernorm(in_0, in_1, in_2, in_3):
    # in_0 = bias [128], in_1 = weight [128]
    # in_2, in_3 = [1, 4, 128]
    N = 128
    num_rows = in_2.numel() // N  # 4
    out = torch.empty_like(in_2)

    fused_add_layernorm_kernel[(num_rows,)](
        in_2, in_3, in_1, in_0, out,
        N=N,
        eps=1e-5,
    )
    return out


def replacement_func():
    return fused_add_layernorm