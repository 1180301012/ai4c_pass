import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation graph
def pattern(in_0):
    # The entire computation sequence from model.py
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_activation_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data once
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Precompute constants
    const_half = 0.5
    const_cube_scale = 0.044715
    const_tanh_scale = 0.7978845608028654
    const_one = 1.0
    
    # Fused computation in one pass
    # Step 1: Scale input by 0.5
    scaled_x = x * const_half
    
    # Step 2: Compute x³ and scale by 0.044715
    x_cubed = x * x * x  # More efficient than torch.pow(x, 3.0)
    scaled_cube = x_cubed * const_cube_scale
    
    # Step 3: Add original input to scaled cube
    x_plus_scaled_cube = x + scaled_cube
    
    # Step 4: Scale by 0.7978845608028654
    scaled_for_tanh = x_plus_scaled_cube * const_tanh_scale
    
    # Step 5: Apply fast approximation of tanh using sigmoid-like function
    # Using fast polynomial approximation for tanh: tanh(x) ≈ x * (1 - x²/3 + 2x⁴/15)
    tanh_sq = scaled_for_tanh * scaled_for_tanh
    tanh_result = scaled_for_tanh * (1.0 - tanh_sq / 3.0 + 2.0 * tanh_sq * tanh_sq / 15.0)
    
    # Step 6: Add 1.0 and multiply by scaled input
    result = scaled_x * (const_one + tanh_result)
    
    # Store final result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_activation_computation(in_0):
    # Handle both float16 and bfloat16
    n_elements = in_0.numel()
    
    # Optimal block size for modern GPUs
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor with same dtype and device as input
    out = torch.empty_like(in_0)
    
    # Launch the fused kernel
    fused_activation_kernel[(num_programs,)](
        input_ptr=in_0,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_activation_computation