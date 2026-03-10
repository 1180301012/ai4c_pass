import torch
import torch.fx
import triton
import triton.language as tl

@torch.fx.wrap
def fused_conv_flatten_transpose(x, weight, bias):
    batch_size, in_channels, hin, win = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    hout = hin - kernel_h + 1
    wout = win - kernel_w + 1
    total_patches = hout * wout
    
    # Create output tensor [batch, patches, channels]
    output = torch.empty(batch_size, total_patches, out_channels, dtype=x.dtype, device=x.device)
    
    # Calculate grid size
    num_programs = batch_size * out_channels
    BLOCK_SIZE = 1024
    
    # Launch kernel
    fused_conv_flatten_transpose_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        hin=hin, win=win,
        kernel_h=kernel_h, kernel_w=kernel_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

@triton.jit
def fused_conv_flatten_transpose_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    hin, win, kernel_h, kernel_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output spatial dimensions
    hout = hin - kernel_h + 1
    wout = win - kernel_w + 1
    total_patches = hout * wout
    
    # Each program handles one batch and one output channel
    pid = tl.program_id(0)
    batch_idx = pid // out_channels
    out_c = pid % out_channels
    
    if batch_idx >= batch_size or out_c >= out_channels:
        return
    
    # Compute the output location in the flattened tensor
    out_offset = batch_idx * out_channels * total_patches + out_c * total_patches
    
    # Load bias for this output channel
    bias_val = tl.load(bias_ptr + out_c)
    
    # Process each spatial location
    for patch_idx in range(total_patches):
        # Calculate spatial coordinates
        hi = patch_idx // wout
        wi = patch_idx % wout
        x_base = batch_idx * in_channels * hin * win
        
        conv_val = bias_val
        
        # Perform convolution
        for ic in range(in_channels):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    x_idx = x_base + ic * hin * win + (hi + kh) * win + (wi + kw)
                    w_idx = out_c * in_channels * kernel_h * kernel_w + ic * kernel_h * kernel_w + kh * kernel_w + kw
                    conv_val += tl.load(x_ptr + x_idx) * tl.load(weight_ptr + w_idx)
        
        # Store result in flattened+transposed format [batch, patches, channels]
        out_idx = out_offset + patch_idx
        tl.store(out_ptr + out_idx, conv_val)

def pattern(x, weight, bias):
    conv_out = torch.conv2d(x, weight, bias, (16, 16), (0, 0), (1, 1), 1)
    flattened = conv_out.flatten(2)
    transposed = flattened.transpose(1, 2)
    return transposed

def replacement_args(x, weight, bias):
    return (x, weight, bias)

def replacement_func():
    return fused_conv_flatten_transpose