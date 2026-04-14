import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    return tmp_4

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def best_fused_kernel(
    sigmoid_ptr,
    input_ptr, 
    output_ptr,
    height,
    width,
    CHANNELS: tl.constexpr,
):
    # One program per spatial location for optimal memory access
    pid = tl.program_id(0)
    x = pid % width
    y = pid // width
    
    # Load all channels efficiently using compile-time arange
    ch_idx = tl.arange(0, CHANNELS)
    
    # Load sigmoid values and compute sigmoid in fp32 precision
    sigmoid_vals = tl.load(sigmoid_ptr + ch_idx)
    sigmoid_fp32 = sigmoid_vals.to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-sigmoid_fp32))
    
    # Compute scaling factor (1 + sigmoid) for fused operation
    scale = 1.0 + sigmoid
    
    # Load input tensor data at this spatial position
    base_idx = ch_idx * height * width + y * width + x
    input_vals = tl.load(input_ptr + base_idx)
    
    # Apply fused optimization: input * (1 + sigmoid) + ReLU
    # This replaces: tmp_2 = in_1 * sigmoid, tmp_3 = in_1 + tmp_2
    result = input_vals * scale
    
    # Apply final ReLU operation
    output_vals = tl.maximum(result, 0.0)
    
    # Store result at same memory location
    tl.store(output_ptr + base_idx, output_vals)

@torch.fx.wrap
def best_optimized_computation(in_0, in_1):
    # Extract tensor dimensions
    _, channels = in_0.shape
    _, _, h, w = in_1.shape
    
    assert channels == 512, f"Expected 512 channels, got {channels}"
    
    # Optimized grid: one CUDA thread per spatial pixel
    grid = (h * w,)
    
    # Prepare output tensor
    output = torch.empty_like(in_1)
    
    # Launch highly optimized kernel
    best_fused_kernel[grid](
        sigmoid_ptr=in_0,
        input_ptr=in_1,
        output_ptr=output,
        height=h,
        width=w,
        CHANNELS=512,
    )
    
    return output

def replacement_func():
    return best_optimized_computation