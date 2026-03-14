import torch
import triton
import triton.language as tl

def pattern(in_8, in_2, in_1, tmp_0):
    # Conv2D operation (in_8, tmp_2, tmp_1)
    tmp_7 = torch.conv2d(in_8, in_2, in_1, (1, 1), (0, 0), (1, 1), 1)
    # Layer scale multiplication 
    tmp_9 = tmp_7 * tmp_0
    return tmp_9

def replacement_args(in_8, in_2, in_1, tmp_0):
    return (in_8, in_2, in_1, tmp_0)

@triton.jit
def simple_conv_layer_scale_kernel(
    input_ptr,
    weight_ptr, 
    bias_ptr,
    gamma_ptr,
    output_ptr,
    N, C_in, H, W, C_out,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Grid mapping: (N, C_out, HW) -> (program_id_0, program_id_1, program_id_2)
    pid_n = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Convert HW index to H, W coordinates
    hw_idx = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    
    # Mask for valid coordinates
    mask = hw_idx < H * W
    h_mask = h_idx < H
    w_mask = w_idx < W
    valid_mask = mask & h_mask & w_mask
    
    # Process each spatial position in the block
    for i in range(BLOCK_SIZE_HW):
        if valid_mask[i]:
            h = h_idx[i]
            w = w_idx[i]
            
            # Process each output channel in this position
            for c_out in range(pid_c_out * BLOCK_SIZE_C, min((pid_c_out + 1) * BLOCK_SIZE_C, C_out)):
                # Load bias and gamma
                bias = tl.load(bias_ptr + c_out)
                gamma = tl.load(gamma_ptr + c_out)
                
                # 1x1 Conv2D computation
                acc = 0.0
                for c_in in range(C_in):
                    # Load weight
                    weight_idx = c_out * C_in + c_in
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Load input value  
                    input_idx = pid_n * C_in * H * W + c_in * H * W + h * W + w
                    input_val = tl.load(input_ptr + input_idx)
                    
                    acc += weight_val * input_val
                
                # Apply bias and layer scale
                output_val = (acc + bias) * gamma
                
                # Store result
                output_idx = pid_n * C_out * H * W + c_out * H * W + h * W + w
                tl.store(output_ptr + output_idx, output_val)

@torch.fx.wrap
def fused_conv_layer_scale(in_8, in_2, in_1, tmp_0):
    input_shape = in_8.shape
    N, C_in, H, W = input_shape
    C_out = in_2.shape[0]
    
    # Output shape
    out_shape = (N, C_out, H, W)
    output = torch.empty(out_shape, dtype=torch.float32, device=in_8.device)
    
    # Optimized block sizes for better GPU occupancy
    BLOCK_SIZE_N = 1  # Process one batch at a time
    BLOCK_SIZE_C = 64  # Process all output channels
    BLOCK_SIZE_HW = 64  # Process 64 spatial positions per thread
    
    # Calculate grid size
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (C_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_hw = (H * W + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    
    simple_conv_layer_scale_kernel[(grid_n, grid_c, grid_hw)](
        in_8,
        in_2,
        in_1,
        tmp_0,
        output,
        N, C_in, H, W, C_out,
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_HW
    )
    
    return output

def replacement_func():
    return fused_conv_layer_scale