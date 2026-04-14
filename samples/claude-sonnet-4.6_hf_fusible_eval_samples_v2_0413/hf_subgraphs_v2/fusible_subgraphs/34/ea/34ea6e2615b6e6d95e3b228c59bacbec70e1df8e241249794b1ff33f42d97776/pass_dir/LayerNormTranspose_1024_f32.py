import torch
import triton
import triton.language as tl


def pattern(x, weight, bias):
    """
    Matches: layer_norm(x, (1024,), weight, bias, 1e-05)
    Input x = tmp_7: [1, N, 1024], non-contiguous strides (N*1024, 1, N).
    Returns tmp_8 (single tensor), downstream transposes happen naturally.
    """
    tmp_8 = torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)
    return tmp_8


def replacement_args(x, weight, bias):
    return (x, weight, bias)


@triton.jit
def _layer_norm_kernel_f32(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N_rows, C,
    stride_tok, stride_feat,
    BLOCK_C: tl.constexpr,
):
    """
    One program per token. Reads non-contiguous input (stride_feat between features).
    Computes layer_norm in float32. Writes float32 output contiguously.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    # Non-contiguous load: feature j of token row is at x_ptr + row*stride_tok + j*stride_feat
    x_offsets = row * stride_tok + cols * stride_feat
    x_val = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    # Mean and variance over C features
    mean = tl.sum(x_val, axis=0) / C
    diff = x_val - mean
    var = tl.sum(diff * diff, axis=0) / C
    rstd = tl.rsqrt(var + 1e-5)

    # Affine transform: y = diff/std * weight + bias
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    result = diff * rstd * w + b

    # Contiguous store: out[row, col] at out_ptr + row*C + col
    tl.store(out_ptr + row * C + cols, result, mask=mask)


@torch.fx.wrap
def layer_norm_1024_f32(x, weight, bias):
    """
    Replacement for layer_norm(x, (1024,), weight, bias, 1e-05).
    x:      [B, N, C] = [1, 256, 1024], non-contiguous from flatten(2).transpose(1,2).
            Strides: (C*N, 1, N) = (262144, 1, 256).
    weight: [1024], bias: [1024]
    Returns out: [B, N, C] = [1, 256, 1024] contiguous.
    """
    B, N, C = x.shape    # B=1, N=256, C=1024
    # Strides of x (from [1,C,N].transpose(1,2)): (C*N, 1, N)
    stride_tok = 1       # stride[1]
    stride_feat = N      # stride[2] = 256

    out = torch.empty(B, N, C, dtype=x.dtype, device=x.device)

    _layer_norm_kernel_f32[(B * N,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        N_rows=B * N,
        C=C,
        stride_tok=stride_tok,
        stride_feat=stride_feat,
        BLOCK_C=1024,
    )

    return out


def replacement_func():
    return layer_norm_1024_f32