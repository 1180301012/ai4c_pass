import torch
import triton
import triton.language as tl

@triton.jit
def conv_sigmoid_mul_kernel(
    sigmoid_out_ptr, # sigmoid output [N, C_out, 1, 1]
    mul_out_ptr,     # multiplication output [N, C_out, H_mul, W_mul]
    x_ptr,           # input tensor [N, C_in, H, W]
    weight_ptr,      # weights [C_out, C_in, K_H, K_W]
    bias_ptr,        # bias [C_out]
    mul_ptr,         # multiplication tensor [N, C_out, H_mul, W_mul]
    N, C_in, C_out, H_in, W_in, H_mul, W_mul,
    K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    batch_idx = pid // (C_out * H_mul * W_mul)
    out_C = (pid // (H_mul * W_mul)) % C_out
    out_h = (pid // W_mul) % H_mul
    out_w = pid % W_mul
    
    # Only process valid indices
    if batch_idx >= N:
        return
    if out_C >= C_out:
        return
    if out_h >= H_mul:
        return
    if out_w >= W_mul:
        return
    
    # For 1x1 convolution with spatial broadcasting
    # Input x has shape [N, C_in, 1, 1], mul has shape [N, C_out, H_mul, W_mul]
    # We need to compute conv2d result at [batch_idx, out_C, 0, 0] then broadcast
    conv_sum = 0.0
    
    # Load bias (it doesn't depend on the loop variable)
    bias_val = tl.load(bias_ptr + out_C)
    
    # Accumulate convolution for the single spatial location
    for c in range(C_in):
        weight_val = tl.load(weight_ptr + out_C * C_in * K_H * K_W + c * K_H * K_W + 0 * K_W + 0)
        x_val = tl.load(x_ptr + batch_idx * C_in * 1 * 1 + c * 1 * 1 + 0 * 1 + 0)
        conv_sum += weight_val * x_val
    
    # Add bias and apply sigmoid
    conv_result = conv_sum + bias_val
    sigmoid_result = 1.0 / (1.0 + tl.exp(-conv_result))
    
    # Multiply with mul tensor and broadcast to output spatial dimensions
    mul_val = tl.load(mul_ptr + batch_idx * C_out * H_mul * W_mul + out_C * H_mul * W_mul + out_h * W_mul + out_w)
    output_val = sigmoid_result * mul_val
    
    # Store result broadcast to all spatial locations
    for h in range(H_mul):
        for w in range(W_mul):
            output_idx = batch_idx * C_out * H_mul * W_mul + out_C * H_mul * W_mul + h * W_mul + w
            tl.store(mul_out_ptr + output_idx, output_val)

@torch.fx.wrap
def fused_conv_sigmoid_mul(x, weight, bias, mul):
    # Get tensor shapes
    N, C_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    H_mul, W_mul = mul.shape[2], mul.shape[3]
    
    # Create output tensor for multiplication result
    # tmp_4: multiplication result [N, C_out, H_mul, W_mul] 
    mul_out = torch.empty_like(mul)
    
    # For 1x1 convolution, we need to handle the spatial broadcasting
    # Input x has shape [N, C_in, 1, 1], mul has shape [N, C_out, H_mul, W_mul]
    # We compute convolution for each batch and channel, then broadcast
    
    BLOCK_SIZE = 256
    grid_size = N * C_out * H_mul * W_mul  # Compute all spatial locations
    num_programs = (grid_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    conv_sigmoid_mul_kernel[(num_programs,)](
        sigmoid_out_ptr=None,  # We don't need sigmoid output
        mul_out_ptr=mul_out,
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        mul_ptr=mul,
        N=N, C_in=C_in, C_out=C_out, 
        H_in=H_in, W_in=W_in, H_mul=H_mul, W_mul=W_mul,
        K_H=1, K_W=1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return mul_out

def pattern(x, weight, bias, mul):
    # Conv2D
    tmp_2 = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    # Set temporaries to None to match original pattern
    tmp_1 = tmp_0 = None
    # Sigmoid
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_2 = None
    # Multiplication
    tmp_4 = mul * tmp_3
    tmp_3 = None
    # Return only the multiplication result (which is used in the next operation)
    return tmp_4

def replacement_args(x, weight, bias, mul):
    return (x, weight, bias, mul)

def replacement_func():
    return fused_conv_sigmoid_mul