import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.gelu(tmp_4, approximate='none')
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    tmp_8 = torch.nn.functional.dropout(tmp_7, 0.0, False, False)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


_BLOCK_COUT = 4


def _prune_configs(configs, named_args, **kwargs):
    hw = named_args['HW']
    valid = [c for c in configs if c.kwargs['BLOCK_HW'] >= hw]
    return valid if valid else configs


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CIN': 64, 'BLOCK_COUT': _BLOCK_COUT}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CIN': 64, 'BLOCK_COUT': _BLOCK_COUT}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_CIN': 64, 'BLOCK_COUT': _BLOCK_COUT}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_CIN': 64, 'BLOCK_COUT': _BLOCK_COUT}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_CIN': 64, 'BLOCK_COUT': _BLOCK_COUT}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_CIN': 64, 'BLOCK_COUT': _BLOCK_COUT}, num_warps=4, num_stages=1),
    ],
    key=['HW'],
    prune_configs_by={'early_config_prune': _prune_configs},
)
@triton.jit
def fused_se_gelu_avgpool_kernel(
    bias_ptr,
    weight_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    C_out,
    C_in,
    HW,
    inv_hw,
    stride_in2_b,
    stride_in2_c,
    num_c_blocks,
    BLOCK_CIN: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    # Each block handles BLOCK_COUT channels for one batch element
    pid = tl.program_id(0)
    b = pid // num_c_blocks
    c_block = pid % num_c_blocks
    c_start = c_block * BLOCK_COUT

    # Load in_3[b, :] once - shared across all output channels
    cin_offs = tl.arange(0, BLOCK_CIN)
    cin_mask = cin_offs < C_in
    x = tl.load(in_3_ptr + b * C_in + cin_offs, mask=cin_mask, other=0.0).to(tl.float32)

    hw_offs = tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW

    # Process channels sequentially (loop is unrolled by compiler)
    for i in tl.static_range(BLOCK_COUT):
        c_out = c_start + i

        # Load weight[c_out, :] and compute dot product
        w = tl.load(weight_ptr + c_out * C_in + cin_offs, mask=cin_mask, other=0.0).to(tl.float32)
        val = tl.sum(x * w, axis=0) + tl.load(bias_ptr + c_out).to(tl.float32)

        # Sigmoid
        se = 1.0 / (1.0 + tl.exp(-val))

        # Load in_2[b, c_out, :], scale, gelu, avgpool
        in2 = tl.load(
            in_2_ptr + b * stride_in2_b + c_out * stride_in2_c + hw_offs,
            mask=hw_mask, other=0.0
        ).to(tl.float32)

        s = in2 * se
        g = s * 0.5 * (1.0 + tl.math.erf(s * 0.7071067811865476))
        g = tl.where(hw_mask, g, 0.0)

        result = tl.sum(g, axis=0) * inv_hw
        tl.store(out_ptr + b * C_out + c_out, result)


@torch.fx.wrap
def fused_se_gelu_avgpool(in_0, in_1, in_2, in_3):
    B = in_3.shape[0]
    C_in = in_3.shape[1]
    C_out = in_1.shape[0]
    H = in_2.shape[2]
    W = in_2.shape[3]
    HW = H * W
    inv_hw = 1.0 / HW

    num_c_blocks = C_out // _BLOCK_COUT

    out = torch.empty((B, C_out), dtype=in_2.dtype, device=in_2.device)

    grid = (B * num_c_blocks,)
    fused_se_gelu_avgpool_kernel[grid](
        in_0, in_1, in_2, in_3, out,
        C_out, C_in, HW, inv_hw,
        C_out * HW, HW,
        num_c_blocks,
    )
    return out


def replacement_func():
    return fused_se_gelu_avgpool