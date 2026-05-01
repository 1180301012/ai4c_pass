import torch
import triton
import triton.language as tl


def pattern(in_9, in_6, in_2, in_3, in_5, in_4):
    conv2d_1 = torch.conv2d(in_9, in_6, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_13 = torch.nn.functional.batch_norm(conv2d_1, in_2, in_3, in_5, in_4, False, 0.1, 1e-05)
    tmp_14 = torch.nn.functional.relu(tmp_13, inplace=False)
    return tmp_14

def replacement_args(in_9, in_6, in_2, in_3, in_5, in_4):
    return (in_9, in_6, in_2, in_3, in_5, in_4)

@triton.jit
def batchnorm_conv_relu_kernel(
    x_ptr,
    w_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    H,
    W,
    C_in,
    C_out,
    BLOCK_SIZE: tl.constexpr
):
    # Process one output element
    i = tl.program_id(0)
    j = tl.program_id(1)
    c_out = tl.program_id(2)

    # Calculate convolution output at (i, j, c_out)
    total = tl.zeros((), dtype=tl.float32)
    
    # 3x3 convolution with padding 1
    for ki in range(3):
        for kj in range(3):
            # Calculate input coordinates with padding
            x_i = i + ki - 1
            x_j = j + kj - 1
            
            # Check bounds
            if x_i < 0 or x_i >= H or x_j < 0 or x_j >= W:
                continue
                
            # Load input
            input_val = tl.load(x_ptr + x_i * W * C_in + x_j * C_in)
            
            # Load kernel weight
            kernel_val = tl.load(w_ptr + c_out * C_in * 9 + (ki * 3 + kj) * C_in)
            
            # Accumulate
            total += input_val * kernel_val

    # Apply batch norm and ReLU
    mean = tl.load(mean_ptr + c_out)
    var = tl.load(var_ptr + c_out)
    weight = tl.load(weight_ptr + c_out)
    bias = tl.load(bias_ptr + c_out)
    
    # Batch norm calculation
    eps = 1e-05
    inv_std = 1.0 / tl.sqrt(var + eps)
    bn_val = (total - mean) * weight * inv_std + bias
    
    # ReLU
    out_val = tl.maximum(0.0, bn_val)
    
    # Store result
    tl.store(out_ptr + i * W * C_out + j * C_out + c_out, out_val)

@torch.fx.wrap
def batchnorm_conv_relu_triton(x, w, mean, var, weight, bias):
    batch, C_in, H, W = x.shape
    C_out, _, _, _ = w.shape
    
    out = torch.empty(batch, C_out, H, W, dtype=x.dtype)
    
    # Configure grid for 3D execution: (H, W, C_out)
    BLOCK_SIZE = 128
    grid = (H, W, C_out)
    
    batchnorm_conv_relu_kernel[grid](
        x_ptr=x,
        w_ptr=w,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        H=H,
        W=W,
        C_in=C_in,
        C_out=C_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return batchnorm_conv_relu_triton