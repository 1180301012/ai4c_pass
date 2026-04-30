import torch
import triton
import triton.language as tl


@triton.jit
def conv_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, out_channels, in_channels, height, width,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    """
    1x1 Convolution kernel with fused reshape.
    Input: [B, IC, H, W]
    Weight: [OC, IC, 1, 1]
    Output: [B, OC, H*W]
    """
    sp_size = height * width
    
    # Program IDs
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)
    
    # Output indices
    oc_idx = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_idx = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    
    # Masks
    oc_mask = oc_idx < out_channels
    sp_mask = sp_idx < sp_size
    
    # Decode spatial index
    h = sp_idx // width
    w = sp_idx % width
    
    # Compute conv: sum over in_channels
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    
    for ic in range(in_channels):
        # Input offset: [B, IC, H, W] -> flat
        inp_off = pid_b * in_channels * sp_size + ic * sp_size + h * width + w
        inp = tl.load(input_ptr + inp_off, mask=sp_mask, other=0.0)
        
        # Weight offset: [OC, IC, 1, 1] -> flat
        w_off = oc_idx * in_channels + ic
        wgt = tl.load(weight_ptr + w_off, mask=oc_mask, other=0.0)
        
        acc += inp * wgt[:, None]
    
    # Add bias
    bias = tl.load(bias_ptr + oc_idx, mask=oc_mask, other=0.0)
    acc += bias[:, None]
    
    # Store output: [B, OC, SP]
    oc_out = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)[:, None]
    sp_out = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)[None, :]
    out_off = (pid_b * out_channels + oc_out) * sp_size + sp_out
    
    store_mask = (oc_out < out_channels) & (sp_out < sp_size)
    tl.store(output_ptr + out_off, acc, mask=store_mask)


@torch.fx.wrap
def fused_conv_reshape(bias, weight, input_tensor):
    """
    Fused 1x1 conv + reshape.
    Input: [B, IC, H, W], Weight: [OC, IC, 1, 1], Bias: [OC]
    Output: [B, OC, H*W]
    """
    # Get dimensions
    batch = input_tensor.shape[0]
    in_channels = weight.shape[1]
    height = input_tensor.shape[2]
    width = input_tensor.shape[3]
    out_channels = weight.shape[0]
    sp_size = height * width
    
    # Output
    output = torch.empty((batch, out_channels, sp_size), 
                         dtype=torch.float32, 
                         device=input_tensor.device)
    
    # Grid
    BLOCK_OC, BLOCK_SP = 16, 64
    grid = (
        batch,
        (out_channels + BLOCK_OC - 1) // BLOCK_OC,
        (sp_size + BLOCK_SP - 1) // BLOCK_SP,
    )
    
    # Launch
    conv_kernel[grid](
        input_tensor, weight, bias, output,
        batch, out_channels, in_channels, height, width,
        BLOCK_OC, BLOCK_SP,
    )
    
    return output.to(input_tensor.dtype)


def pattern(in_0, in_1, in_2):
    """
    Match: conv2d + mul(1.0) + reshape
    The mul by 1.0 is a no-op that can be eliminated.
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv_reshape