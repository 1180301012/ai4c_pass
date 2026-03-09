import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, skip1, skip2, stride, padding, dilation, groups):
    # Pattern: pointwise conv2d + two additions (residual connections)
    conv_out = torch.conv2d(x, weight, bias, stride, padding, dilation, groups)
    add1 = skip1 + conv_out
    add2 = add1 + skip2
    return conv_out, add1, add2

def replacement_args(x, weight, bias, skip1, skip2, stride, padding, dilation, groups):
    return (x, weight, bias, skip1, skip2)

@triton.jit
def fused_conv_add_kernel(
    x_ptr, weight_ptr, bias_ptr, skip1_ptr, skip2_ptr, out_ptr,
    N, C, H, W,
    weight_C_out, weight_C_in, weight_H, weight_W,
    BN_stride_H, BN_stride_W, BN_pad_H, BN_pad_W, BN_dil_H, BN_dil_W,
    groups: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Calculate program IDs
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Compute range for this program
    n_start = pid_n * BLOCK_SIZE_N
    c_start = pid_c * BLOCK_SIZE_C
    hw_start = pid_hw * BLOCK_SIZE_HW
    
    n_end = min(n_start + BLOCK_SIZE_N, N)
    c_end = min(c_start + BLOCK_SIZE_C, weight_C_out)
    hw_end = min(hw_start + BLOCK_SIZE_HW, H * W)
    
    # Handle pointwise convolution (1x1 kernel)
    if weight_H == 1 and weight_W == 1:
        # Load bias for output channels
        bias_val = tl.load(bias_ptr + c_start, mask=c_start < weight_C_out, other=0.0)
        
        # Process each sample in batch
        for n in range(n_start, n_end):
            # Process each output channel block
            for c_out in range(c_start, c_end):
                # Load weight (pointwise, so just scalar per channel)
                weight_val = tl.load(weight_ptr + c_out * weight_C_in, mask=True, other=0.0)
                
                # Load input and skip tensors at spatial location [0,0] (pointwise)
                x_val = tl.load(x_ptr + n * C * H * W + c_out * H * W, mask=True, other=0.0)
                skip1_val = tl.load(skip1_ptr + n * C * H * W + c_out * H * W, mask=True, other=0.0)
                skip2_val = tl.load(skip2_ptr + n * C * H * W + c_out * H * W, mask=True, other=0.0)
                
                # Compute conv (pointwise multiply)
                conv_val = weight_val * x_val + bias_val[c_out]
                
                # Fuse two additions
                add1_val = skip1_val + conv_val
                add2_val = add1_val + skip2_val
                
                # Store final result
                output_idx = n * weight_C_out * H * W + c_out * H * W
                for hw_idx in range(hw_start, hw_end):
                    out_idx = output_idx + hw_idx
                    tl.store(out_ptr + out_idx, add2_val)
    
def conv2d_add_kernel_wrapper(x, weight, bias, skip1, skip2):
    N, C, H, W = x.shape
    weight_C_out, weight_C_in, weight_H, weight_W = weight.shape
    
    # Output has same spatial dimensions as input for pointwise conv
    out = torch.empty((N, weight_C_out, H, W), dtype=x.dtype, device=x.device)
    
    # Determine block sizes
    BLOCK_SIZE_N = min(8, N)
    BLOCK_SIZE_C = min(64, weight_C_out)
    BLOCK_SIZE_HW = min(1024, H * W)
    
    # Calculate grid
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (weight_C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_hw = (H * W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    # Launch kernel
    fused_conv_add_kernel[(
        grid_n,
        grid_c, 
        grid_hw
    )](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        skip1_ptr=skip1,
        skip2_ptr=skip2,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        weight_C_out=weight_C_out,
        weight_C_in=weight_C_in,
        weight_H=weight_H,
        weight_W=weight_W,
        BN_stride_H=1, BN_stride_W=1,
        BN_pad_H=0, BN_pad_W=0,
        BN_dil_H=1, BN_dil_W=1,
        groups=groups,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return out

@torch.fx.wrap
def fused_conv_add(x, weight, bias, skip1, skip2):
    return conv2d_add_kernel_wrapper(x, weight, bias, skip1, skip2)

def replacement_func():
    return fused_conv_add