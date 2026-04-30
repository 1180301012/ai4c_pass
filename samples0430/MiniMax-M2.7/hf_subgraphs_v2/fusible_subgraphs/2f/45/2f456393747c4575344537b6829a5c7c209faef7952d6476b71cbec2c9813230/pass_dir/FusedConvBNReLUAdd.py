import torch
import triton
import triton.language as tl

# Pattern matching function - matches Conv2D + BatchNorm + LeakyReLU + Add pattern
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Match the computation pattern:
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    tmp_8 = tmp_7 + in_5
    return (tmp_8,)
    """
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    tmp_8 = tmp_7 + in_5
    return tmp_8

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Extract arguments needed for the fused kernel:
    - in_0: running_mean (BN)
    - in_1: running_var (BN)
    - in_2: bias (BN)
    - in_3: weight (BN)
    - in_4: conv_weight
    - in_5: residual (add input)
    - in_6: input tensor
    """
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@triton.jit
def fused_conv_bn_relu_add_kernel(
    # Input and output pointers
    x_ptr, y_ptr, residual_ptr,
    # Conv weight: [out_channels, in_channels, kH, kW] (contiguous)
    weight_ptr,
    # BN parameters: [out_channels]
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    # Dimensions
    N, C_out, H_out, W_out,  # Output dimensions [batch, channels, height, width]
    C_in, H_in, W_in,  # Input dimensions
    # BN fusion parameters
    eps: tl.constexpr,
    # LeakyReLU parameter
    neg_slope: tl.constexpr,
    # Output dtype as constexpr for proper casting
    out_dtype: tl.constexpr,
    # Performance tuning
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Conv2D + BatchNorm + LeakyReLU + Add (residual)
    Optimized for stride=1, padding=1, groups=1 (3x3 conv)
    
    The kernel handles padding by checking bounds for each load.
    """
    pid = tl.program_id(0)
    
    # Calculate total output elements
    total_out = N * C_out * H_out * W_out
    
    # Bounds check - exit early if this program has no work
    if pid >= (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE:
        return
    
    # Calculate position within this program's block
    block_start = pid * BLOCK_SIZE
    local_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = local_offsets < total_out
    
    # Decode position into N, C, H, W coordinates for OUTPUT
    n = (local_offsets // (C_out * H_out * W_out)) % N
    c_out = (local_offsets // (H_out * W_out)) % C_out
    h_out = (local_offsets // W_out) % H_out
    w_out = local_offsets % W_out
    
    # Compute convolution: accumulate over input channels and kernel (3x3 with padding=1)
    conv_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Convolution loop for 3x3 kernel with stride=1, padding=1
    # With padding=1 and original input size (H_in), the effective input range is [-1, H_in]
    # We check bounds for each load and use 0 for out-of-bounds positions
    for c_in in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # Input position with padding: h_in = h_out + kh - 1 (range -1 to H_in)
                h_in = h_out + kh - 1
                w_in = w_out + kw - 1
                
                # Check bounds: valid range is [0, H_in-1] and [0, W_in-1]
                h_valid = (h_in >= 0) & (h_in < H_in)
                w_valid = (w_in >= 0) & (w_in < W_in)
                valid = h_valid & w_valid
                
                # Compute flat index only for valid positions
                # Use select to handle the condition: if valid, use computed index, else 0
                in_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in
                
                # Load with bounds check: only load if valid, otherwise 0
                x_val = tl.load(x_ptr + in_idx, mask=valid & mask, other=0.0)
                
                # Load weight: [out_channels, in_channels, kH, kW]
                # PyTorch default memory layout: (out, in, kH, kW)
                weight_idx = ((c_out * C_in + c_in) * 3 + kh) * 3 + kw
                w_val = tl.load(weight_ptr + weight_idx)
                
                # Accumulate convolution
                conv_out = conv_out + x_val * w_val
    
    # Apply fused BatchNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    # Load BN parameters and cast to float32 for computation
    bn_mean = tl.load(bn_mean_ptr + c_out).to(tl.float32)
    bn_var = tl.load(bn_var_ptr + c_out).to(tl.float32)
    bn_weight = tl.load(bn_weight_ptr + c_out).to(tl.float32)
    bn_bias = tl.load(bn_bias_ptr + c_out).to(tl.float32)
    
    # BN scale factor (in float32)
    bn_scale = bn_weight / tl.sqrt(bn_var + eps)
    
    # Apply fused BN
    bn_out = (conv_out - bn_mean) * bn_scale + bn_bias
    
    # Apply LeakyReLU
    relu_out = tl.where(bn_out > 0, bn_out, bn_out * neg_slope)
    
    # Load residual and add
    # Residual has same shape as output: [N, C_out, H_out, W_out]
    res_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out
    residual = tl.load(residual_ptr + res_idx, mask=mask, other=0.0).to(tl.float32)
    
    final_out = relu_out + residual
    
    # Cast result to output dtype
    final_out = final_out.to(out_dtype)
    
    # Store result
    tl.store(y_ptr + local_offsets, final_out, mask=mask)


@torch.fx.wrap
def fused_conv_bn_relu_add_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Wrapper for the fused Conv + BatchNorm + LeakyReLU + Add kernel.
    """
    # Get dimensions from input tensors
    N, C_in, H_in, W_in = in_6.shape
    C_out = in_4.shape[0]
    kH, kW = in_4.shape[2], in_4.shape[3]
    
    # For stride=1, padding=1, kernel=3: output size equals input size
    H_out = H_in
    W_out = W_in
    
    # BN parameters
    eps = 1e-05
    
    # Output dtype - determine from input
    out_dtype = in_6.dtype
    if out_dtype == torch.float16:
        triton_dtype = tl.float16
    elif out_dtype == torch.bfloat16:
        triton_dtype = tl.bfloat16
    elif out_dtype == torch.float32:
        triton_dtype = tl.float32
    else:
        triton_dtype = tl.float32  # default
    
    # Output tensor
    y = torch.empty((N, C_out, H_out, W_out), device=in_6.device, dtype=in_6.dtype)
    
    # Calculate total output elements
    total_out = N * C_out * H_out * W_out
    BLOCK_SIZE = 128
    
    # Ensure tensors are contiguous and on the right device
    x = in_6.contiguous()
    w = in_4.contiguous()
    residual = in_5.contiguous()
    bn_mean = in_0.contiguous()
    bn_var = in_1.contiguous()
    bn_weight = in_3.contiguous()
    bn_bias = in_2.contiguous()
    
    # Launch kernel
    num_programs = (total_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv_bn_relu_add_kernel[(num_programs,)](
        x_ptr=x, y_ptr=y, residual_ptr=residual,
        weight_ptr=w,
        bn_mean_ptr=bn_mean, bn_var_ptr=bn_var, bn_weight_ptr=bn_weight, bn_bias_ptr=bn_bias,
        N=N, C_out=C_out, H_out=H_out, W_out=W_out,
        C_in=C_in, H_in=H_in, W_in=W_in,
        eps=eps,
        neg_slope=0.01,
        out_dtype=triton_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y


def replacement_func():
    """Return the fused kernel wrapper function"""
    return fused_conv_bn_relu_add_wrapper