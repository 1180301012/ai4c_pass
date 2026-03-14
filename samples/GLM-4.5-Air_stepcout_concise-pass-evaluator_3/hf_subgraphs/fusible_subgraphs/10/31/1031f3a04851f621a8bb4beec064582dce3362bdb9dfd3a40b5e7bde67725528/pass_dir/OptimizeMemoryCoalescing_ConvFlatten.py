import torch
import triton
import triton.language as tl

# Pattern matching function (same as before for consistency)
def pattern(in_0, in_1, in_2):
    """Pattern: Conv2D with 1x1 kernel followed by flatten along spatial dimensions"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(tmp_2, 2)
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the optimized convolution and flatten operation"""
    return (in_0, in_1, in_2)

# Optimized kernel with memory coalescing improvements
@triton.jit
def optimized_conv_flatten_kernel(
    input_ptr,      # [B, C_in, H, W]
    weight_ptr,     # [C_out, C_in, 1, 1] 
    bias_ptr,       # [C_out]
    output_ptr,     # [B, C_out, H*W]
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    """Optimized kernel with improved memory coalescing and batch processing"""
    # Optimized grid indexing for better memory coalescing
    pid = tl.program_id(0)
    spatial_idx = pid % width
    batch_group = (pid // width) // BLOCK_BATCH
    batch_idx = (pid // width) % BLOCK_BATCH
    out_ch_group = batch_group // (height * BLOCK_BATCH)
    out_ch_idx = (batch_group % (height * BLOCK_BATCH)) // BLOCK_BATCH
    y = (batch_group % (height * BLOCK_BATCH)) % height
    
    # Compute actual indices
    batch_idx_actual = batch_idx + batch_group * BLOCK_BATCH
    out_ch_idx_actual = out_ch_idx + out_ch_group * BLOCK_BATCH
    
    if batch_idx_actual >= batch_size or out_ch_idx_actual >= out_channels or spatial_idx >= width:
        return
    
    x = spatial_idx
    
    # Optimized memory access pattern with coalescing
    result = 0.0
    offset = tl.arange(0, BLOCK_SIZE)
    
    # Process channels in chunks for memory coalescing
    for ch_base in range(0, in_channels, BLOCK_SIZE):
        in_bounds = ch_base + offset < in_channels
        
        # Weight access pattern optimized for coalescing weights
        weights = tl.load(weight_ptr + out_ch_idx_actual * in_channels + ch_base + offset, 
                         mask=in_bounds, other=0.0)
        
        # Input access pattern optimized for coalescing input data
        # Group batch and channel accesses for better memory locality
        input_offset = (ch_base + offset) * height * width + y * width + x
        inputs = tl.load(input_ptr + batch_idx_actual * in_channels * height * width + input_offset, 
                         mask=in_bounds, other=0.0)
        
        # Optimized accumulation
        result += tl.sum(weights * inputs)
    
    # Bias addition with coalesced access
    bias = tl.load(bias_ptr + out_ch_idx_actual)
    final_result = result + bias
    
    # Output storage with coalesced pattern
    output_idx = batch_idx_actual * out_channels * width + out_ch_idx_actual * width + spatial_idx
    tl.store(output_ptr + output_idx, final_result)

@torch.fx.wrap
def optimized_conv_flatten_forward(bias, weight, input_tensor):
    """Wrapper function with enhanced memory coalescing optimization"""
    B, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    spatial_size = H * W
    
    # Output shape: [B, C_out, H*W]
    output = torch.empty((B, C_out, spatial_size), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # Enhanced block size selection for memory coalescing
    workload_product = B * C_out
    
    if workload_product > 1000:
        # Large workloads: optimized for memory coalescing
        BLOCK_SIZE = 128
        BLOCK_BATCH = 4
    elif workload_product > 100:
        # Medium workloads: balanced coalescing and parallelism
        BLOCK_SIZE = 64
        BLOCK_BATCH = 4
    else:
        # Small workloads: minimize overhead
        BLOCK_SIZE = 64
        BLOCK_BATCH = 1
    
    # Calculate optimized grid size for memory coalescing
    total_outputs = B * C_out * W
    total_programs = (total_outputs + BLOCK_BATCH * W - 1) // (BLOCK_BATCH * W)
    
    # Launch kernel with memory coalescing optimization
    optimized_conv_flatten_kernel[(total_programs,)](
        input_tensor,
        weight, 
        bias,
        output,
        B,
        C_in,
        C_out,
        H,
        W,
        spatial_size,
        BLOCK_SIZE,
        BLOCK_BATCH
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_conv_flatten_forward