import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['C'],
)
@triton.jit
def fused_relu_bn_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, C)

    # Load BN statistics (shared across all rows, cached in L2)
    mean = tl.load(mean_ptr + col_offsets).to(tl.float32)
    var  = tl.load(var_ptr  + col_offsets).to(tl.float32)
    w    = tl.load(weight_ptr + col_offsets).to(tl.float32)
    b    = tl.load(bias_ptr  + col_offsets).to(tl.float32)

    # Precompute inverse std
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Load input row
    x = tl.load(input_ptr + row * C + col_offsets)
    input_dtype = x.dtype

    # ReLU in fp32
    x_f32 = tl.maximum(x.to(tl.float32), 0.0)

    # Fused BatchNorm: (relu(x) - mean) * inv_std * weight + bias
    x_out = (x_f32 - mean) * inv_std * w + b

    # Store output cast back to original dtype
    tl.store(output_ptr + row * C + col_offsets, x_out.to(input_dtype))


@torch.fx.wrap
def fused_relu_bn_dropout(in_0, in_1, in_2, in_3, in_4):
    """
    Fused ReLU + BatchNorm (inference, training=False) + Dropout(p=0.0)
    Args:
        in_0: running_mean  [C]  (on CPU)
        in_1: running_var   [C]  (on CPU)
        in_2: bias          [C]  (on CPU)
        in_3: weight        [C]  (on CPU)
        in_4: input tensor  [N, C] (on CUDA)
    Returns:
        output tensor [N, C] (on CUDA)
    """
    N, C = in_4.shape
    device = in_4.device

    # Move BN statistics to the same device as input
    mean   = in_0.to(device)
    var    = in_1.to(device)
    weight = in_3.to(device)
    bias   = in_2.to(device)

    output = torch.empty_like(in_4)

    # One program per row
    grid = (N,)

    fused_relu_bn_kernel[grid](
        in_4, mean, var, weight, bias, output,
        C, 1e-05,
    )

    return output


def replacement_func():
    return fused_relu_bn_dropout