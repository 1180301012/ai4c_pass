import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Match layer_norm + transpose pattern
    tmp_2 = torch.nn.functional.layer_norm(in_2, (768,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-1, -2)
    return (tmp_3,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_layer_norm_transpose_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Compute program index
    m = tl.program_id(0)
    n_start = tl.program_id(1) * BLOCK_SIZE_N
    
    # Initialize shared memory for weight and bias
    weight_shared = tl.shared.alloc([hidden_size], dtype=tl.float32)
    bias_shared = tl.shared.alloc([hidden_size], dtype=tl.float32)
    
    # Load weight and bias to shared memory
    weight_offset = tl.arange(0, hidden_size)
    weight_mask = weight_offset < hidden_size
    tl.store(weight_shared + weight_offset, tl.load(weight_ptr + weight_offset, mask=weight_mask), mask=weight_mask)
    
    bias_offset = tl.arange(0, hidden_size)
    bias_mask = bias_offset < hidden_size
    tl.store(bias_shared + bias_offset, tl.load(bias_ptr + bias_offset, mask=bias_mask), mask=bias_mask)
    
    # Synchronize to ensure shared memory is loaded
    tl.sync()
    
    # Load input data
    input_offsets = m * seq_len * hidden_size + n_start + tl.arange(0, BLOCK_SIZE_N)
    mask = n_start + tl.arange(0, BLOCK_SIZE_N) < hidden_size
    input_data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Apply layer normalization
    weight_data = tl.load(weight_shared + n_start + tl.arange(0, BLOCK_SIZE_N), mask=mask, other=1.0)
    bias_data = tl.load(bias_shared + n_start + tl.arange(0, BLOCK_SIZE_N), mask=mask, other=0.0)
    
    # Layer normalization computation
    eps = 1e-05
    input_centered = input_data - tl.mean(input_data)
    variance = tl.mean(input_centered * input_centered)
    std_inv = 1.0 / tl.sqrt(variance + eps)
    
    output_normed = input_centered * std_inv
    output_final = output_normed * weight_data + bias_data
    
    # Store result (without transpose first, then we'll handle the transpose)
    tl.store(output_ptr + input_offsets, output_final, mask=mask)





@torch.fx.wrap
def fused_layer_norm_transpose_kernel_wrapper_simple(in_0, in_1, in_2):
    # Get input shapes - we know hidden_size is always 768 from weight metadata
    batch_size = in_2.size(0)
    seq_len = in_2.size(1) 
    hidden_size = 768  # From weight metadata: [768] for both weight and bias
    
    # Create output tensor with transposed dimensions
    output = torch.empty((batch_size, hidden_size, seq_len), dtype=in_2.dtype, device=in_2.device)
    
    # Optimized block sizes for better performance - using power-of-2 arange requirement
    BLOCK_SIZE_HIDDEN = 256  # Power-of-2 for better vectorization
    
    # Calculate grid dimensions for 3D parallelization
    grid_x = batch_size           # One program per batch element
    grid_y = seq_len              # One program per sequence position  
    grid_z = (hidden_size + BLOCK_SIZE_HIDDEN - 1) // BLOCK_SIZE_HIDDEN  # Blocks for hidden dimension
    
    # Launch simple optimized kernel
    grid = (grid_x, grid_y, grid_z)
    
    fused_layer_norm_transpose_kernel[grid](
        in_2,
        in_1,
        in_0,
        output,
        batch_size,
        seq_len,
        hidden_size,
        BLOCK_SIZE_HIDDEN,
        BLOCK_SIZE_HIDDEN  # Use same size for both block dimensions
    )
    
    return output

def replacement_func():
    return fused_layer_norm_transpose_kernel_wrapper_simple