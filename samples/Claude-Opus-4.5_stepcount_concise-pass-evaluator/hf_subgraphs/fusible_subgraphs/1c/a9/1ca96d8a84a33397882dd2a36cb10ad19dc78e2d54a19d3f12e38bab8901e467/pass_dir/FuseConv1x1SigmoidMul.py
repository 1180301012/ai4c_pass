import torch
import triton
import triton.language as tl


def pattern(bias, weight, x_28, x_34):
    """
    Pattern to match conv2d (1x1) + sigmoid + broadcast multiply
    This matches the spatial attention pattern in LiteHRNet
    """
    conv_out = torch.conv2d(x_34, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    sig_out = torch.sigmoid(conv_out)
    mul_out = x_28 * sig_out
    return mul_out


def replacement_args(bias, weight, x_28, x_34):
    return (bias, weight, x_28, x_34)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['BATCH', 'HW'],
)
@triton.jit
def fused_conv1x1_sigmoid_mul_kernel(
    bias_ptr, weight_ptr, x_28_ptr, x_34_ptr, out_ptr,
    BATCH, C_OUT, HW,
    stride_x28_b, stride_x28_c,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused kernel for 1x1 conv + sigmoid + broadcast multiply
    
    Grid: (BATCH * C_OUT, num_hw_blocks)
    """
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    b = pid_bc // C_OUT
    c_out = pid_bc % C_OUT
    
    # C_IN = 10, pad to 16 for efficient access
    c_in_offs = tl.arange(0, 16)
    c_in_mask = c_in_offs < 10
    
    # Load and compute dot product
    x_vals = tl.load(x_34_ptr + b * 10 + c_in_offs, mask=c_in_mask, other=0.0)
    w_vals = tl.load(weight_ptr + c_out * 10 + c_in_offs, mask=c_in_mask, other=0.0)
    acc = tl.sum(x_vals * w_vals, axis=0)
    
    # Add bias and sigmoid
    bias_val = tl.load(bias_ptr + c_out)
    sig_out = tl.sigmoid(acc + bias_val)
    
    # Broadcast multiply
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < HW
    
    idx = b * stride_x28_b + c_out * stride_x28_c + hw_offsets
    x28_vals = tl.load(x_28_ptr + idx, mask=hw_mask, other=0.0)
    out_vals = x28_vals * sig_out
    
    out_idx = b * C_OUT * HW + c_out * HW + hw_offsets
    tl.store(out_ptr + out_idx, out_vals, mask=hw_mask)


@torch.fx.wrap
def fused_conv1x1_sigmoid_mul(bias, weight, x_28, x_34):
    """
    Fused implementation of conv1x1 + sigmoid + broadcast multiply
    """
    BATCH = x_28.shape[0]
    C_OUT = x_28.shape[1]
    H = x_28.shape[2]
    W = x_28.shape[3]
    HW = H * W
    
    out = torch.empty_like(x_28)
    
    weight_flat = weight.view(C_OUT, -1).contiguous()
    x_34_flat = x_34.view(BATCH, -1).contiguous()
    
    # Use 256 as base block size
    BLOCK_HW = 256
    num_hw_blocks = triton.cdiv(HW, BLOCK_HW)
    
    grid = (BATCH * C_OUT, num_hw_blocks)
    
    stride_x28_b = x_28.stride(0)
    stride_x28_c = x_28.stride(1)
    
    fused_conv1x1_sigmoid_mul_kernel[grid](
        bias, weight_flat, x_28, x_34_flat, out,
        BATCH, C_OUT, HW,
        stride_x28_b, stride_x28_c,
    )
    
    return out


def replacement_func():
    return fused_conv1x1_sigmoid_mul