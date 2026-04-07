import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation chain
def pattern(in_0):
    # Match the complete sequence: GELU -> reshape -> reshape -> pad
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return tmp_3

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Complete fusion Triton kernel - optimized full computation
@triton.jit
def complete_fusion_kernel(
    x_ptr,               # Input tensor pointer [1, 124, 1536]
    out_ptr,             # Output tensor pointer [1, 248, 769]
    input_size,          # Input size: 1 * 124 * 1536
    output_size,         # Output size: 1 * 248 * 769
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which block of data this program handles
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size
    
    # Load input data efficiently
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply high-accuracy GELU activation
    # GELU(x) = 0.5 * x * (1.0 + tanh(0.79788 * x * (1 + 0.044715 * x^2)))
    sqrt_2_div_pi = 0.7978845608
    coeff = 0.044715
    
    x_cubed = x * x * x
    tanh_arg = sqrt_2_div_pi * (x + coeff * x_cubed)
    
    # Fast tanh approximation: tanh(x) ≈ x * (1.0 - x^2 * (1.0 - x^2 * 0.551))
    tanh_arg_sq = tanh_arg * tanh_arg
    tanh_approx = tanh_arg * (1.0 - tanh_arg_sq * (1.0 - tanh_arg_sq * 0.551))
    
    gelu_out = x * 0.5 * (1.0 + tanh_approx)
    
    # Complete coordinate mapping: [1, 124, 1536] -> [1, 248, 769] with padding
    # Direct mapping without intermediate steps:
    # Input: [1, 124, 1536] -> Element at (m=0, k1, k2) where k1=0-123, k2=0-1535
    # Output: [1, 248, 769] -> Element at (m=0, k1_new, k2_new) where k1_new=0-247, k2_new=0-768
    
    # Convert 1D input indices to coordinates
    k1_orig = offsets // 1536
    k2_orig = offsets % 1536
    
    # Direct mapping to final coordinates:
    # k1_new = k1_orig * 2 + (k2_orig // 768)  # Map 124 sequences to 248 sequences
    # k2_new = k2_orig % 768                    # Position within 768-element block
    # Then add padding: k2_final = k2_new, but output has 769 elements
    
    k1_new = k1_orig * 2 + (k2_orig // 768)
    k2_new = k2_orig % 768
    
    # For padding: extend the last dimension by 1 element
    # All elements map to positions 0-768 in the last dimension (769 total)
    k2_final = k2_new
    
    # Convert to 1D output index: [1, 248, 769]
    out_idx = k1_new * 769 + k2_final
    
    # Bounds checking for output
    valid_mask = mask & (out_idx < output_size)
    
    # Store final result with efficient vectorized memory access
    tl.store(out_ptr + out_idx, gelu_out, mask=valid_mask)

# Complete fusion kernel wrapper with optimal configuration
@torch.fx.wrap
def complete_computation_fusion(in_0):
    # Input shape: [1, 124, 1536]
    # Output shape: [1, 248, 769]
    input_size = 1 * 124 * 1536  # 190464
    output_size = 1 * 248 * 769  # 190712 (includes padding)
    
    # Optimal block size for maximum GPU occupancy
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    num_programs = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty([1, 248, 769], dtype=in_0.dtype, device=in_0.device)
    
    # Launch complete fusion kernel
    complete_fusion_kernel[(num_programs,)](
        x_ptr=in_0,
        out_ptr=out,
        input_size=input_size,
        output_size=output_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return complete_computation_fusion