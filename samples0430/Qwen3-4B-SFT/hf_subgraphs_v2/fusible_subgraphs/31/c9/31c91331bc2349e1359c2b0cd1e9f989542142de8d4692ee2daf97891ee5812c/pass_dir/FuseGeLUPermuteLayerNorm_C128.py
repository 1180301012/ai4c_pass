import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=2),
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 128}, num_warps=16),
    ],
    key=['S'],
)
@triton.jit
def _fused_kernel_c128(
    in2_ptr,
    in3_ptr,
    weight_ptr,
    bias_ptr,
    out2_ptr,
    out12_ptr,
    C: tl.constexpr,
    S,
    BLOCK_C: tl.constexpr,
):
    s = tl.program_id(0)
    c_offs = tl.arange(0, BLOCK_C)
    valid = c_offs < C

    # Load in_2[0, c, s] (memory layout: in_2 treated as [C, S])
    x = tl.load(in2_ptr + c_offs * S + s, mask=valid, other=0.0)
    x_f32 = x.to(tl.float32)

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    xsqrt2 = x_f32 * 0.7071067811865476
    gelu = 0.5 * xsqrt2 * (1.0 + tl.math.erf(xsqrt2))

    # Load in_3[0, s, c]
    r = tl.load(in3_ptr + s * C + c_offs, mask=valid, other=0.0)
    r_f32 = r.to(tl.float32)

    # Residual add
    z = gelu + r_f32

    # Store out2 (tmp_10): [S, C] layout (swapped from original permute chain)
    tl.store(out2_ptr + s * C + c_offs, z.to(x.dtype), mask=valid)

    # Layer norm over C dimension
    z_m = tl.where(valid, z, 0.0)
    mean = tl.sum(z_m, axis=0) / C
    diff = tl.where(valid, z - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C
    inv_std = 1.0 / tl.sqrt(var + 1e-6)
    z_norm = diff * inv_std

    w = tl.load(weight_ptr + c_offs, mask=valid, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c_offs, mask=valid, other=0.0).to(tl.float32)
    out_f32 = z_norm * w + b

    # Store out12 (tmp_12): [H*W, C] layout = [S, C]
    tl.store(out12_ptr + s * C + c_offs, out_f32.to(x.dtype), mask=valid)


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    tmp_7 = tmp_6.permute(0, 2, 1)
    tmp_8 = tmp_7.view(1, 128, 16, 12)
    tmp_9 = tmp_8.view(1, 128, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (128,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 16, 12, 128)
    return (tmp_10, tmp_12)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def _fused_fn_c128(in_0, in_1, in_2, in_3):
    # in_0: bn bias  [C]
    # in_1: bn weight [C]
    # in_2: gelu input  [1, C, H, W]
    # in_3: residual    [1, S, C]
    C = 128
    S = in_2.shape[2] * in_2.shape[3]   # H * W

    out2 = torch.empty(in_3.shape, dtype=in_3.dtype, device=in_3.device)
    out12 = torch.empty(1, S, C, dtype=in_3.dtype, device=in_3.device)

    _fused_kernel_c128[(S,)](
        in_2, in_3, in_1, in_0, out2, out12,
        C=C, S=S,
    )

    return (out2, out12.view(1, 16, 12, 128))


def replacement_func():
    return _fused_fn_c128