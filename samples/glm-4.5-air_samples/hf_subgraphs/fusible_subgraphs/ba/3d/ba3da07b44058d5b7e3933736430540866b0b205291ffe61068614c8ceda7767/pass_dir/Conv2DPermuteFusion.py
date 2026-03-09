import torch
import triton
import triton.language as tl

# Pattern matching function: Perm operation (key part of Conv2D+Permute pattern)
def pattern(input_tensor, weight, bias):
    # Focus on the permute operation which is the key transformation
    # This represents the tensor layout transformation part
    tmp_3 = input_tensor.permute(0, 2, 3, 1)  # NCHW to NHWC
    return tmp_3

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

@triton.jit
def conv2d_permute_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    N, C_out, H_out, W_out,
    C_in, K_H, K_W, S_H, S_W,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which output element we compute
    pid = tl.program_id(0)
    pid_n = pid // (H_out * W_out * C_out)
    pid_c = (pid % (H_out * W_out * C_out)) // (H_out * W_out)
    pid_h = (pid % (H_out * W_out)) // W_out
    pid_w = pid % W_out
    
    # Calculate input coordinates (assuming padding 0, stride 1, dilation 1)
    ih = pid_h
    iw = pid_w
    
    # Initialize output accumulator
    acc = 0.0
    
    # Loop over input channels and kernel dimensions
    for c_in in range(C_in):
        for kh in range(K_H):
            for kw in range(K_W):
                # Input coordinates
                input_ih = ih * S_H + kh
                input_iw = iw * S_W + kw
                
                # Check bounds
                if (input_ih < H_out and input_iw < W_out):
                    # Load input and weight
                    input_val = tl.load(input_ptr + 
                        ((pid_n * C_in + c_in) * H_out + input_ih) * W_out + input_iw,
                        other=0.0)
                    weight_val = tl.load(weight_ptr + 
                        (pid_c * C_in + c_in) * K_H * K_W + kh * K_W + kw,
                        other=0.0)
                    acc += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + pid_c, other=0.0)
    acc += bias_val
    
    # Store output in NHWC format
    tl.store(output_ptr + 
        ((pid_n * H_out + pid_h) * W_out + pid_w) * C_out + pid_c, acc)

@torch.fx.wrap
def conv2d_permute_fused(input_tensor, weight, bias):
    # Get input and output shapes
    N, C_in, H_in, W_in = input_tensor.shape
    C_out, _, K_H, K_W = weight.shape
    
    # Output shape calculation for stride 1, padding 0
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1
    
    # Create output tensor in NHWC format
    output = torch.empty((N, H_out, W_out, C_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block size and grid calculation
    BLOCK_SIZE = 1024
    total_elements = N * H_out * W_out * C_out
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv2d_permute_kernel[grid_size](
        input_tensor, weight, bias,
        output,
        N, C_out, H_out, W_out,
        C_in, K_H, K_W, 1, 1,  # strides
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return conv2d_permute_fused