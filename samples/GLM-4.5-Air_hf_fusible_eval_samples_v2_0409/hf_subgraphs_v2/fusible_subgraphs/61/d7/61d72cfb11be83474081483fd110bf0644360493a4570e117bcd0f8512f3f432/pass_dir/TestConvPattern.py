import torch
import triton
import triton.language as tl

def pattern(weight, input_tensor):
    # Exact match for conv2d with same parameters
    conv_result = torch.conv2d(input_tensor, weight, None, (1, 1), (0, 0), (1, 1), 1)
    return conv_result

def replacement_args(weight, input_tensor):
    return (weight, input_tensor)

@triton.jit
def simple_conv_kernel(
    input_ptr, weight_ptr, output_ptr,
    N, C_in, H, W, C_out,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    m = (pid // BLOCK_SIZE_N) * BLOCK_SIZE_M
    n = (pid % BLOCK_SIZE_N)
    
    # Calculate output position for this thread
    batch = n // (C_out * W)
    c_out = (n % (C_out * W)) // W
    w_pos = n % W
    h_pos = pid % (C_out * BLOCK_SIZE_N) * BLOCK_SIZE_N // W  # Simplified for 1D grid
    
    if h_pos < H:
        if c_out < C_out:
            if batch < N:
                # Compute 1x1 convolution: sum over input channels
                sum_val = 0.0
                for c_in in range(C_in):
                    # Load input value
                    input_offset = batch * C_in * H * W + c_in * H * W + h_pos * W + w_pos
                    input_val = tl.load(input_ptr + input_offset)
                    
                    # Load weight 
                    weight_offset = c_out * C_in + c_in
                    weight_val = tl.load(weight_ptr + weight_offset)
                    
                    sum_val += input_val * weight_val
                
                # Store output
                output_offset = batch * C_out * H * W + c_out * H * W + h_pos * W + w_pos
                tl.store(output_ptr + output_offset, sum_val)

@torch.fx.wrap
def simple_conv_replacement(weight, input_tensor):
    N, C_in, H, W = input_tensor.shape
    C_out, _, _, _ = weight.shape
    
    output = torch.zeros([N, C_out, H, W], dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Simple grid for 1x1 conv - one thread per batch and spatial position
    total_elements = N * H * W
    grid_size = total_elements
    
    simple_conv_kernel[(grid_size,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        output_ptr=output,
        N=N, C_in=C_in, H=H, W=W, C_out=C_out,
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=1,
    )
    
    return output

def replacement_func():
    return simple_conv_replacement