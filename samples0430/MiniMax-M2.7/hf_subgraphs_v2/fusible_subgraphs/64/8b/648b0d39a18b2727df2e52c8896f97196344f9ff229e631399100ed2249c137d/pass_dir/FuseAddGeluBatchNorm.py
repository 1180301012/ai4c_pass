import torch
import triton
import triton.language as tl

@triton.jit
def fused_gelu_bn_kernel(
    x_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_gelu_ptr, out_bn_ptr,
    n_elements,
    n_channels,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes BLOCK_SIZE elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU using erf approximation: x * 0.5 * (1 + erf(x / sqrt(2)))
    # Cast to float32 for erf which doesn't support fp16/bf16, then cast back
    x_fp32 = x.to(tl.float32)
    gelu_fp32 = x_fp32 * 0.5 * (1.0 + tl.math.erf(x_fp32 * 0.7071067811865476))
    gelu_out = gelu_fp32.to(x.type)
    
    # Store gelu output (tmp_5)
    tl.store(out_gelu_ptr + offsets, gelu_out, mask=mask)
    
    # Load batch norm parameters for channel (using modulo for channel indexing)
    channel_offsets = offsets % n_channels
    mean = tl.load(mean_ptr + channel_offsets, mask=mask)
    var_val = tl.load(var_ptr + channel_offsets, mask=mask)
    w = tl.load(weight_ptr + channel_offsets, mask=mask)
    b = tl.load(bias_ptr + channel_offsets, mask=mask)
    
    # Compute inv_std = 1 / sqrt(var + eps) using rsqrt for efficiency
    inv_std = tl.math.rsqrt(var_val + eps)
    
    # Normalize: (x - mean) / sqrt(var + eps) * weight + bias
    bn_out = (gelu_out - mean) * inv_std * w + b
    
    # Store bn output (tmp_6)
    tl.store(out_bn_ptr + offsets, bn_out, mask=mask)


@torch.fx.wrap
def fused_add_gelu_bn_kernel_wrapper(
    in_0, in_1, in_2, in_3, in_4, in_5
):
    """
    Fused kernel: add + gelu + batch_norm
    
    in_0: running mean (C,)
    in_1: running variance (C,)
    in_2: batch norm bias (C,)
    in_3: batch norm weight (C,)
    in_4: input tensor (B, C, H, W)
    in_5: input tensor to add (B, C, H, W)
    """
    # Compute the sum: in_4 + in_5
    x = in_4 + in_5
    
    # Flatten to 1D for simplicity
    n_elements = x.numel()
    n_channels = x.shape[1]
    
    # Allocate outputs
    out_gelu = torch.empty_like(x)
    out_bn = torch.empty_like(x)
    
    # Configure grid
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel - inv_std is computed inside the kernel using rsqrt
    fused_gelu_bn_kernel[(num_programs,)](
        x_ptr=x,
        mean_ptr=in_0, var_ptr=in_1, weight_ptr=in_3, bias_ptr=in_2,
        out_gelu_ptr=out_gelu, out_bn_ptr=out_bn,
        n_elements=n_elements,
        n_channels=n_channels,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_gelu, out_bn


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the complete computation pattern from the model:
    1. in_4 += in_5
    2. tmp_5 = gelu(in_4)
    3. tmp_6 = batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    4. tmp_7 = 0 + tmp_6
    Returns: (tmp_5, tmp_7)
    """
    # In-place add (matches model's in_4 += in_5)
    in_4 += in_5
    
    # GELU activation
    tmp_5 = torch.nn.functional.gelu(in_4, approximate='none')
    
    # Batch normalization
    tmp_6 = torch.nn.functional.batch_norm(
        tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05
    )
    
    # Identity add (0 + tmp_6)
    tmp_7 = 0 + tmp_6
    
    return (tmp_5, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Extract arguments for the fused kernel.
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    """
    Returns the fused kernel function.
    """
    return fused_add_gelu_bn_kernel_wrapper