import inspect
import operator
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused Triton kernel: avg_pool2d(2x2,stride=2) + BN (inference) + SiLU
# Variant 2: uses aten.batch_norm.default (returns single tensor, no getitem)
#            + aten.silu_.default  (inplace)
#
# Input layout: [1, 512, 16, 16]  (contiguous)
# Output shape: [1, 512, 8, 8]
# ---------------------------------------------------------------------------

@triton.jit
def fused_avgpool_bn_silu_v2_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    eps,
    H_IN:        tl.constexpr,
    W_IN:        tl.constexpr,
    W_OUT:       tl.constexpr,
    SPATIAL_OUT: tl.constexpr,
    IS_BF16:     tl.constexpr,
):
    pid_c = tl.program_id(0)

    mean_val   = tl.load(mean_ptr   + pid_c)
    var_val    = tl.load(var_ptr    + pid_c)
    weight_val = tl.load(weight_ptr + pid_c)
    bias_val   = tl.load(bias_ptr   + pid_c)

    inv_std = 1.0 / tl.sqrt(var_val + eps)
    scale   = weight_val * inv_std
    shift   = bias_val - mean_val * scale

    s_offs = tl.arange(0, SPATIAL_OUT)
    h_out  = s_offs // W_OUT
    w_out  = s_offs %  W_OUT

    h0 = h_out * 2
    w0 = w_out * 2

    ch_base = pid_c * H_IN * W_IN

    idx00 = ch_base + h0       * W_IN + w0
    idx01 = ch_base + h0       * W_IN + w0 + 1
    idx10 = ch_base + (h0 + 1) * W_IN + w0
    idx11 = ch_base + (h0 + 1) * W_IN + w0 + 1

    x00 = tl.load(input_ptr + idx00).to(tl.float32)
    x01 = tl.load(input_ptr + idx01).to(tl.float32)
    x10 = tl.load(input_ptr + idx10).to(tl.float32)
    x11 = tl.load(input_ptr + idx11).to(tl.float32)

    pooled = (x00 + x01 + x10 + x11) * 0.25
    bn     = pooled * scale + shift
    silu   = bn * tl.sigmoid(bn)

    if IS_BF16:
        result = silu.to(tl.bfloat16)
    else:
        result = silu.to(tl.float16)

    out_base = pid_c * SPATIAL_OUT
    tl.store(output_ptr + out_base + s_offs, result)


@torch.fx.wrap
def fused_avgpool_bn_silu_v2(x, running_mean, running_var, weight, bias):
    """
    Fused: avg_pool2d(2,2) + batch_norm(inference) + silu.
    x: [1, 512, 16, 16] on CUDA.
    """
    device = x.device
    dtype  = x.dtype
    x_c    = x.contiguous()

    C           = 512
    H_IN = W_IN = 16
    H_OUT = W_OUT = 8
    SPATIAL_OUT  = H_OUT * W_OUT   # 64
    eps          = 1e-05

    rmean = running_mean.to(device=device, dtype=torch.float32)
    rvar  = running_var .to(device=device, dtype=torch.float32)
    w     = weight      .to(device=device, dtype=torch.float32)
    b     = bias        .to(device=device, dtype=torch.float32)

    output  = torch.empty(1, C, H_OUT, W_OUT, device=device, dtype=dtype)
    is_bf16 = (dtype == torch.bfloat16)

    fused_avgpool_bn_silu_v2_kernel[(C,)](
        x_c, rmean, rvar, w, b,
        output, eps,
        H_IN=H_IN, W_IN=W_IN, W_OUT=W_OUT, SPATIAL_OUT=SPATIAL_OUT,
        IS_BF16=is_bf16,
        num_warps=2,
    )
    return output


# ---------------------------------------------------------------------------
# Pattern v2: 3-op chain (avg_pool2d + batch_norm + silu), no reshape.
# x matches the reshape output node [1, 512, 16, 16].
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, x):
    """3-op subgraph: avg_pool2d → batch_norm(inference) → silu(inplace)."""
    tmp_5 = torch.nn.functional.avg_pool2d(x, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(
        tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05
    )
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7


pattern = torch.fx.symbolic_trace(pattern)
pattern.__signature__ = inspect.Signature([
    inspect.Parameter("in_0", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("in_1", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("in_2", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("in_3", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("x",    inspect.Parameter.POSITIONAL_OR_KEYWORD),
])


def replacement_args(in_0, in_1, in_2, in_3, x):
    # x → reshape output [1, 512, 16, 16]
    return (x, in_0, in_1, in_3, in_2)


def replacement_func():
    return fused_avgpool_bn_silu_v2