import torch
import triton
import triton.language as tl


@triton.jit
def fused_mul_bn_kernel(
    x_ptr,
    sig_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for: mul(sigmoid, x) -> batch_norm"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.load(sig_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    scaled = x * sig
    
    # Calculate channel index: (offset // (height * width)) % n_channels
    hw = height * width
    channel_idx = (offsets // hw) % n_channels
    
    mean = tl.load(mean_ptr + channel_idx).to(tl.float32)
    var = tl.load(var_ptr + channel_idx).to(tl.float32)
    weight = tl.load(weight_ptr + channel_idx).to(tl.float32)
    bias = tl.load(bias_ptr + channel_idx).to(tl.float32)
    
    var_eps = var + eps
    inv_std = 1.0 / tl.sqrt(var_eps)
    normalized = (scaled - mean) * inv_std
    output = normalized * weight + bias
    
    tl.store(out_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def fused_mul_bn_impl(x, sig, mean, var, weight, bias, eps=1e-05, training=False):
    """Fused mul + batch_norm"""
    B, C, H, W = x.shape
    
    n_elements = B * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    fused_mul_bn_kernel[(num_programs,)](
        x_ptr=x,
        sig_ptr=sig,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        n_elements=n_elements,
        n_channels=C,
        height=H,
        width=W,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match pattern: sigmoid * x -> batch_norm
    """
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # The pattern returns batch_norm output, which has the same shape as input
    # batch_norm args: input, running_mean, running_var, weight, bias
    # in_0=mean, in_1=var, in_2=bias, in_3=weight, in_4=sigmoid, in_5=input
    # Replacement expects: x, sig, mean, var, weight, bias
    return (in_5, in_4, in_0, in_1, in_3, in_2)


def replacement_func():
    return fused_mul_bn_impl