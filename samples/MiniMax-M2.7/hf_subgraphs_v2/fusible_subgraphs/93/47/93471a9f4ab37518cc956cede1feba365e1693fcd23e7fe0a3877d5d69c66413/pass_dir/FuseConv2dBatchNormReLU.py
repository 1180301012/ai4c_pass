import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_bn_relu_kernel(
    input_ptr, weight_ptr, running_mean_ptr, running_var_ptr,
    weight_bn_ptr, bias_bn_ptr,
    output_ptr,
    # Conv params
    in_channels, out_channels, kernel_h, kernel_w,
    in_h, in_w, out_h, out_w,
    # BN params
    bn_eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get position
    pid = tl.program_id(0)
    num_blocks = out_channels * out_h * out_w
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    # Calculate output indices
    c = offsets // (out_h * out_w)
    h = (offsets % (out_h * out_w)) // out_w
    w = offsets % out_w
    h_actual = h + 3  # padding for 7x7 kernel with stride 1
    w_actual = w + 3
    
    # Load running stats
    mean = tl.load(running_mean_ptr + c)
    var = tl.load(running_var_ptr + c)
    inv_var = tl.rsqrt(var + bn_eps)
    gamma = tl.load(weight_bn_ptr + c)
    beta = tl.load(bias_bn_ptr + c)
    
    # Depthwise conv - each output channel corresponds to one input channel
    conv_out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Unroll the 7x7 kernel
    for kh in range(7):
        for kw in range(7):
            ih = h_actual - kh
            iw = w_actual - kw
            in_mask = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)
            
            # Input position: (c, ih, iw), linear index: c * in_h * in_w + ih * in_w + iw
            in_idx = c * in_h * in_w + ih * in_w + iw
            weight_idx = c * 7 * 7 + kh * 7 + kw
            
            inp = tl.load(input_ptr + in_idx, mask=in_mask, other=0.0)
            wgt = tl.load(weight_ptr + weight_idx)
            conv_out += inp * wgt
    
    # BatchNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    bn_out = (conv_out - mean) * inv_var * gamma + beta
    
    # ReLU: max(0, x)
    bn_out = tl.where(bn_out > 0, bn_out, 0.0)
    
    # Store output
    out_idx = c * out_h * out_w + h * out_w + w
    tl.store(output_ptr + out_idx, bn_out, mask=mask)


@torch.fx.wrap
def fused_conv_bn_relu_wrapper(
    in_0, in_1, in_2, in_3, in_4, in_9
):
    """
    Fused kernel: Conv2d (depthwise) + BatchNorm + ReLU
    
    Inputs:
    - in_0: running_mean [512]
    - in_1: running_var [512]  
    - in_2: bias [512]
    - in_3: weight [512]
    - in_4: conv_weight [512, 1, 7, 7]
    - in_9: input [1, 512, 70, 70]
    
    Output: [1, 512, 64, 64]
    """
    batch, in_channels, in_h, in_w = in_9.shape
    out_channels = 512
    kernel_h = kernel_w = 7
    out_h = in_h - kernel_h + 1  # 64
    out_w = in_w - kernel_w + 1  # 64
    
    # BN parameters
    bn_eps = 1e-05
    
    # Prepare tensors
    input_flat = in_9.reshape(-1)
    weight_flat = in_4.reshape(-1)
    output = torch.empty((batch, out_channels, out_h, out_w), 
                         dtype=in_9.dtype, device=in_9.device)
    output_flat = output.reshape(-1)
    
    # Grid: one program per output element
    num_elements = out_channels * out_h * out_w
    BLOCK_SIZE = 128
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv_bn_relu_kernel[(num_programs,)](
        input_ptr=input_flat,
        weight_ptr=weight_flat,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        weight_bn_ptr=in_3,
        bias_bn_ptr=in_2,
        output_ptr=output_flat,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        bn_eps=bn_eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
    """
    Match the pattern: conv2d -> view -> batch_norm -> relu -> cat
    Returns the intermediate result and the cat output.
    """
    conv2d = torch.conv2d(input=in_9, weight=in_4, groups=512)
    tmp_5 = conv2d.view(1, 512, 64, 64)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.relu(tmp_6, inplace=False)
    tmp_8 = torch.cat([in_5, in_7, in_8, in_6, tmp_7], dim=1)
    return tmp_7, tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9)


def replacement_func():
    return fused_conv_bn_relu_wrapper