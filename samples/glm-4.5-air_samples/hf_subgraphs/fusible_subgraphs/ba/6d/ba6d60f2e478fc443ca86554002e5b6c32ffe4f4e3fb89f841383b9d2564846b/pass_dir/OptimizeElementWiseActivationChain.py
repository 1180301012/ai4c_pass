import torch
import triton
import triton.language as tl

def pattern(in_6, tmp_7):
    # Element-wise multiplication followed by ReLU
    # This matches the pattern: mul -> relu
    tmp_8 = in_6 * tmp_7
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=True)
    return tmp_9

def replacement_args(in_6, tmp_7):
    return (in_6, tmp_7)

@triton.jit
def fused_mul_relu_kernel(
    input1_ptr,  # in_6: [N, C, H, W]
    input2_ptr,  # tmp_7: [N, C_out, 1, 1] (to be broadcasted)
    output_ptr,  # output: [N, C, H, W]
    N, C, H, W, C_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    pid_n = tl.program_id(3)
    
    # Calculate input pointer offsets
    input1_offset = pid_n * C * H * W + pid_c * H * W + pid_h * W + pid_w
    
    # Load first input value
    x = tl.load(input1_ptr + input1_offset, other=0.0)
    
    # Load second input value (broadcasted from 1x1 to full spatial size)
    # Since input2 is [N, C_out, 1, 1], we need to map the first C channels
    # This assumes C >= C_out and we're using the first C_out channels
    if pid_c < C_out:
        input2_offset = pid_n * C_out * 1 * 1 + pid_c * 1 * 1
        y = tl.load(input2_ptr + input2_offset, other=0.0)
    else:
        y = 1.0  # If channel exceeds C_out, use 1.0 for multiplication
    
    # Apply multiplication followed by ReLU (fused)
    # z = relu(x * y) = max(x * y, 0.0)
    result = x * y
    z = tl.maximum(result, 0.0)
    
    # Store result
    tl.store(output_ptr + input1_offset, z)

@triton.jit
def optimized_mul_relu_kernel(
    input1_ptr,  # in_6: [N, C, H, W]
    input2_ptr,  # tmp_7: [N, C, H, W] (full spatial size for optimized version)
    output_ptr,  # output: [N, C, H, W]
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element  
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    pid_n = tl.program_id(3)
    
    # Calculate input pointer offsets
    input1_offset = pid_n * C * H * W + pid_c * H * W + pid_h * W + pid_w
    input2_offset = pid_n * C * H * W + pid_c * H * W + pid_h * W + pid_w
    
    # Load input values
    x = tl.load(input1_ptr + input1_offset, other=0.0)
    y = tl.load(input2_ptr + input2_offset, other=0.0)
    
    # Apply multiplication followed by ReLU (fused)
    # z = relu(x * y) = max(x * y, 0.0)
    result = x * y
    z = tl.maximum(result, 0.0)
    
    # Store result
    tl.store(output_ptr + input1_offset, z)

@torch.fx.wrap
def optimized_element_wise_activation(in_6, tmp_7):
    # Check if we can use the optimized version (both tensors same spatial size)
    N1, C1, H1, W1 = in_6.shape
    N2, C2, H2, W2 = tmp_7.shape
    
    if H1 == H2 and W1 == W2:
        # Use optimized version that processes all elements directly
        N, C, H, W = N1, C1, H1, W1
        
        # Create output tensor
        output = torch.empty_like(in_6)
        
        # Kernel launch parameters
        BLOCK_SIZE = 1024  # Elements to process per program
        
        # Grid size: (H, W, C, N)
        grid = (
            (H + 31) // 32,  # H blocks
            (W + 31) // 32,  # W blocks
            (C + 31) // 32,  # C blocks
            N                # N blocks
        )
        
        # Launch optimized kernel
        optimized_mul_relu_kernel[grid](
            in_6,
            tmp_7,
            output,
            N, C, H, W,
            BLOCK_SIZE
        )
        
        return output
    else:
        # Use broadcasted version
        N, C, H, W = N1, C1, H1, W1
        C_out = C2
        
        # Create output tensor
        output = torch.empty_like(in_6)
        
        # Kernel launch parameters  
        BLOCK_SIZE = 1024  # Elements to process per program
        
        # Grid size: (H, W, C, N)
        grid = (
            (H + 31) // 32,  # H blocks
            (W + 31) // 32,  # W blocks
            (C + 31) // 32,  # C blocks
            N                # N blocks
        )
        
        # Launch broadcasted kernel
        fused_mul_relu_kernel[grid](
            in_6,
            tmp_7,
            output,
            N, C, H, W, C_out,
            BLOCK_SIZE
        )
        
        return output

def replacement_func():
    return optimized_element_wise_activation