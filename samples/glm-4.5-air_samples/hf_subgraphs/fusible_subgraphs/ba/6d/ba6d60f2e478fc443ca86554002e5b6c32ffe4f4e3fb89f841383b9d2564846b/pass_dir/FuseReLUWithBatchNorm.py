import torch
import triton
import triton.language as tl

def pattern(tmp_9, tmp_0, tmp_1, tmp_3, tmp_2):
    # ReLU followed by BatchNorm fusion
    # This matches the pattern: relu -> batch_norm
    output = torch.nn.functional.batch_norm(tmp_9, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    return output

def replacement_args(tmp_9, tmp_0, tmp_1, tmp_3, tmp_2):
    return (tmp_9, tmp_0, tmp_1, tmp_3, tmp_2)

@triton.jit
def fused_relu_batchnorm_kernel(
    input_ptr,        # tmp_9: [N, C, H, W]
    running_mean_ptr, # tmp_0: [C]
    running_var_ptr,  # tmp_1: [C] 
    weight_ptr,       # tmp_3: [C]
    bias_ptr,         # tmp_2: [C]
    output_ptr,       # output: [N, C, H, W]
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2) 
    pid_n = tl.program_id(3)
    
    # Calculate input pointer offset
    input_offset = pid_n * C * H * W + pid_c * H * W + pid_h * W + pid_w
    
    # Load normalization parameters for this channel
    mean = tl.load(running_mean_ptr + pid_c, other=0.0)
    var = tl.load(running_var_ptr + pid_c, other=1.0)
    weight = tl.load(weight_ptr + pid_c, other=1.0)
    bias = tl.load(bias_ptr + pid_c, other=0.0)
    
    # Load input value
    x = tl.load(input_ptr + input_offset, other=0.0)
    
    # Apply batch norm + relu fused operation
    # First: batch norm
    denominator = tl.sqrt(var + eps)
    y = (x - mean) * weight / denominator + bias
    
    # Then: relu (fused)
    z = tl.maximum(y, 0.0)
    
    # Store result
    tl.store(output_ptr + input_offset, z)

@torch.fx.wrap
def fused_relu_batchnorm(tmp_9, tmp_0, tmp_1, tmp_3, tmp_2):
    # Get input shapes
    N, C, H, W = tmp_9.shape
    
    # Create output tensor
    output = torch.empty_like(tmp_9)
    
    # Kernel launch parameters
    eps = 1e-05  # Epsilon from original batch norm
    BLOCK_SIZE = 1024  # Elements to process per program
    
    # Grid size: (H, W, C, N)
    grid = (
        (H + 31) // 32,  # H blocks (32 elements per block)
        (W + 31) // 32,  # W blocks  
        (C + 31) // 32,  # C blocks
        N                # N blocks
    )
    
    # Launch kernel
    fused_relu_batchnorm_kernel[grid](
        tmp_9,
        tmp_0,
        tmp_1, 
        tmp_3,
        tmp_2,
        output,
        N, C, H, W,
        eps,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_relu_batchnorm