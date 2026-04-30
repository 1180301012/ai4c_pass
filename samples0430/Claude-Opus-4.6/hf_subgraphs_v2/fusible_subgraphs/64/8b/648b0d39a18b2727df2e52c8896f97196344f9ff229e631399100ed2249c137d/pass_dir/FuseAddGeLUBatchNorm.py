import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    in_4 += in_5
    tmp_5 = torch.nn.functional.gelu(in_4, approximate='none')
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = 0 + tmp_6
    return (tmp_5, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['HW'],
)
@triton.jit
def fused_add_gelu_bn_kernel(
    in4_ptr, in5_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_gelu_ptr, out_bn_ptr,
    C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (spatial_blocks_per_slice, N*C)
    slice_id = tl.program_id(1)
    block_id = tl.program_id(0)

    # Channel index from slice
    c = slice_id % C
    base_offset = slice_id * HW

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    # Load input data
    data_offsets = base_offset + offsets
    x = tl.load(in4_ptr + data_offsets, mask=mask).to(tl.float32)
    y = tl.load(in5_ptr + data_offsets, mask=mask).to(tl.float32)

    # Add
    sum_val = x + y

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu_out = 0.5 * sum_val * (1.0 + tl.math.erf(sum_val * 0.7071067811865476))

    # BatchNorm (eval mode): weight * (input - mean) / sqrt(var + eps) + bias
    bn_mean = tl.load(mean_ptr + c).to(tl.float32)
    bn_var = tl.load(var_ptr + c).to(tl.float32)
    bn_weight = tl.load(weight_ptr + c).to(tl.float32)
    bn_bias = tl.load(bias_ptr + c).to(tl.float32)

    inv_std = tl.math.rsqrt(bn_var + 1e-5)
    bn_out = bn_weight * (gelu_out - bn_mean) * inv_std + bn_bias

    # Store both outputs
    tl.store(out_gelu_ptr + data_offsets, gelu_out, mask=mask)
    tl.store(out_bn_ptr + data_offsets, bn_out, mask=mask)


@torch.fx.wrap
def fused_add_gelu_bn(in_0, in_1, in_2, in_3, in_4, in_5):
    # in_0: running_mean [C]
    # in_1: running_var [C]
    # in_2: bias [C]
    # in_3: weight [C]
    # in_4: input tensor [N, C, H, W]
    # in_5: input tensor [N, C, H, W]

    shape = in_4.shape
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]
    HW = H * W
    NC = N * C

    out_gelu = torch.empty_like(in_4)
    out_bn = torch.empty_like(in_4)

    grid = lambda META: ((HW + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'], NC)

    fused_add_gelu_bn_kernel[grid](
        in_4, in_5,
        in_0, in_1, in_3, in_2,  # mean, var, weight, bias
        out_gelu, out_bn,
        C, HW,
    )

    return (out_gelu, out_bn)


def replacement_func():
    return fused_add_gelu_bn