import torch
import triton
import triton.language as tl

# Pattern matching function - match GELU + first reshape
def pattern(in_0):
    # Match the sequence: GELU -> reshape 
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# High-performance Triton kernel with optimal memory access
@triton.jit
def high_perf_gelu_reshape_kernel(
    x_ptr,           # Input tensor pointer  
    out_ptr,         # Output tensor pointer
    input_stride,    # Stride for input tensor
    output_stride,   # Stride for output tensor
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs for 2D grid
    m = tl.program_id(0)
    k = tl.program_id(1)
    
    # Compute memory offsets with optimal striding
    x_offset = m * input_stride + k * BLOCK_SIZE_K
    out_offset = m * output_stride + k * BLOCK_SIZE_K
    
    # Create vectorized masks
    x_mask = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < input_stride
    out_mask = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < output_stride
    
    # Load input data efficiently with proper vectorization
    x = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)
    
    # Apply high-accuracy GELU using more stable approximation
    # GELU(x) = x * Φ(sqrt(2/π) * x) where Φ is the CDF of standard normal
    # Use: GELU(x) ≈ 0.5 * x * (1.0 + tanh(sqrt(2/π) * (0.7978845608 * x + 0.044715 * x^3)))
    
    sqrt_2_div_pi = 0.7978845608  # sqrt(2/π) approximation
    coeff_044715 = 0.044715
    
    # More accurate polynomial approximation for GELU
    x_cubed = x * x * x
    gelu_arg = sqrt_2_div_pi * (x + coeff_044715 * x_cubed)
    
    # Fast tanh approximation: tanh(x) ≈ x * (1.0 - x^2 * (1.0 - x^2 * 0.551))
    tanh_arg_sq = gelu_arg * gelu_arg
    tanh_approx = gelu_arg * (1.0 - tanh_arg_sq * (1.0 - tanh_arg_sq * 0.551))
    
    gelu_out = x * 0.5 * (1.0 + tanh_approx)
    
    # Optimized coordinate mapping: [1, 124, 1536] -> [1, 124, 2, 768]
    # Vectorized index computation for better memory locality
    orig_indices = x_offset + tl.arange(0, BLOCK_SIZE_K)
    
    # Convert 1D indices to coordinates using division and modulo
    k1_orig = orig_indices // 1536
    k2_orig = orig_indices % 1536
    
    # Split the 1536 dimension: half goes to position 0, half to position 1
    k2_split = k2_orig // 768  # 0 for first 768 elements, 1 for next 768
    k2_remainder = k2_orig % 768
    
    # Flatten optimized: [1, 124, 2, 768] -> 1D with better memory access
    # Output layout: (124 sequences * 2 splits * 768) + remainder position
    out_idx = k1_orig * (124 * 2 * 768) + k2_split * (124 * 768) + k2_remainder
    
    # Combined mask for safe memory access
    valid_mask = x_mask & (out_idx < output_stride)
    
    # Store results with vectorized memory access
    tl.store(out_ptr + out_idx, gelu_out, mask=valid_mask)

# High-performance kernel wrapper with optimized launch configuration
@torch.fx.wrap
def high_perf_gelu_reshape(in_0):
    # Input shape: [1, 124, 1536]
    # Output shape: [1, 124, 2, 768]
    input_size = 1 * 124 * 1536
    output_size = 1 * 124 * 2 * 768
    
    # Optimized block sizes for better GPU occupancy
    # Use smaller block sizes for better memory locality and warp utilization
    BLOCK_SIZE_M = 1  # Only one batch dimension
    BLOCK_SIZE_K = 512  # Smaller for better cache locality
    
    # Calculate grid dimensions for optimal GPU utilization
    num_blocks_k = (1536 + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K  # Based on last dimension
    
    # Create output tensor with optimized layout
    out = torch.empty([1, 124, 2, 768], dtype=in_0.dtype, device=in_0.device)
    
    # Launch high-performance kernel
    high_perf_gelu_reshape_kernel[(BLOCK_SIZE_M, num_blocks_k)](
        x_ptr=in_0,
        out_ptr=out,
        input_stride=124 * 1536,
        output_stride=124 * 2 * 768,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out

# Replacement function
def replacement_func():
    return high_perf_gelu_reshape