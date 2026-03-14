import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """Match the conv2d -> interpolate -> final sigmoid sequence"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, size=(640, 640), mode='bilinear')
    tmp_9 = torch.nn.functional.sigmoid(tmp_3)
    return (tmp_9,)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """Extract arguments needed for the fused kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def optimized_sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized sigmoid kernel using tanh approximation for better performance"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized sigmoid using: sigmoid(x) = 0.5 * (1 + tanh(x/2))
    # This is mathematically equivalent but often faster than direct exp calculation
    x_scaled = x * 0.5
    tanh_val = tl.tanh(x_scaled)
    sigmoid = 0.5 * (1.0 + tanh_val)
    
    # Store result
    tl.store(output_ptr + offsets, sigmoid, mask=mask)

@torch.fx.wrap
def optimized_conv_interpolate_sigmoid(in_0, in_1, in_2):
    """Optimized function for conv2d + interpolate + sigmoid sequence"""
    
    # Step 1: Perform the conv2d operation (keep as-is since it's already optimized)
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), 1)
    
    # Step 2: Perform the interpolate operation (keep as-is since PyTorch is optimized)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, size=(640, 640), mode='bilinear')
    
    # Step 3: Apply optimized sigmoid to the interpolated result
    n_elements = tmp_3.numel()
    
    # Create output tensor
    tmp_9 = torch.empty_like(tmp_3)
    
    # Launch optimized sigmoid kernel
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_sigmoid_kernel[(grid_size,)](
        tmp_3,
        tmp_9,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_9

def replacement_func():
    """Return the optimized conv-interpolate-sigmoid function"""
    return optimized_conv_interpolate_sigmoid