import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_view_kernel_batch32(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N, C, H, W,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)
    
    # Calculate output coordinates (after conv = [32, 1, 20, 20])
    out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    out_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    
    # Load weight [1, 64, 1, 1] - only the relevant channel
    weight_offset = (out_c[0]) * 1 * 1
    weight = tl.load(weight_ptr + weight_offset, mask=out_c[0] < 1)
    
    # Load bias [1]
    bias = tl.load(bias_ptr + 0)
    
    # Process each input location [32, 64, 20, 20] -> [32, 1, 20, 20]
    for i in range(H):
        for j in range(W):
            # Conv2D with 1x1 kernel, stride 1, padding 0
            in_h = i
            in_w = j
            
            # Load input value for each batch and channel
            for n in range(BLOCK_SIZE_N):
                if out_n[n] < N:
                    # Input offset for [N, C, H, W] layout
                    input_offset = (out_n[n] * C * H * W) + (out_c[0] * H * W) + (in_h * W) + in_w
                    x = tl.load(input_ptr + input_offset, mask=(out_n[n] < N) & (out_c[0] < C))
                    
                    # Apply conv2d: x * weight + bias
                    conv_val = x * weight + bias
                    
                    # Store result in [N, C_out, H_out, W_out] = [32, 1, 20, 20]
                    out_offset = (out_n[n] * 1 * 20 * 20) + (out_c[0] * 20 * 20) + out_h[n] * 20 + out_w[n]
                    tl.store(out_ptr + out_offset, conv_val, mask=(out_n[n] < N) & (out_h[n] < 20) & (out_w[n] < 20))

@torch.fx.wrap
def conv2d_view_optimized_batch32(N_batch, C_in, H_in, W_in, input_tensor, weight_tensor, bias_tensor):
    # Output after conv2d: [32, 1, 20, 20]
    C_out = 1
    H_out = 20
    W_out = 20
    
    # Set grid dimensions
    grid = (
        (N_batch + 7) // 8,  # N dimension blocks (32 batches)
        (H_out + 15) // 16,  # H dimension blocks
        (W_out + 15) // 16,  # W dimension blocks  
        (C_out + 15) // 16,  # C dimension blocks
    )
    
    # Create output tensor
    output = torch.empty((N_batch, 1, H_out, W_out), dtype=torch.float32, device="cuda")
    
    # Launch kernel
    conv2d_view_kernel_batch32[grid](
        input_tensor,
        weight_tensor.flatten(),  # weight is [1, 64, 1, 1]
        bias_tensor,
        output,
        N_batch, C_in, H_in, W_in,
    )
    
    # Apply view: [32, 1, 20, 20] -> [32, 1, 400]
    return output.view(32, 1, 400)

def pattern(conv_input, conv_weight, conv_bias):
    # Conv2D operation
    conv_output = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # View operation  
    view_output = conv_output.view(32, 1, -1)
    
    return conv_output, view_output

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return conv2d_view_optimized_batch32