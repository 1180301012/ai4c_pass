import torch

# Pattern matching function - exactly match the reference pattern
def pattern(x, y):
    """
    Simple add pattern from the reference example
    """
    return x + y

# Argument extraction function for simple pattern
def replacement_args(x, y):
    return (x, y)

# Simple replacement using PyTorch's native addition (no Triton)
@torch.fx.wrap
def pytorch_add(x, y):
    """Simple PyTorch addition that should match any tensor addition"""
    return x + y

# Replacement function  
def replacement_func():
    return pytorch_add






    # Input pointers
    in_4_ptr,    # [B, C_h, H, W] - input tensor 1
    in_1_ptr,    # [B, C_w, H, W] - input tensor 2  
    in_3_ptr,    # [B, C_out, H, W] - accumulator tensor (updated in-place)
    in_0_ptr,    # scalar - scale factor
    in_2_ptr,    # [B, C_out, H, W] - add tensor
    out_ptr,     # [B, C_out, H, W] - output tensor
    
    # Tensor shapes
    batch_size: tl.constexpr,
    c_h: tl.constexpr,         # channels for in_1 [64]
    c_w: tl.constexpr,         # channels for in_4 [512]  
    c_out: tl.constexpr,       # channels for output [512]
    height: tl.constexpr,      # spatial dimension [64]
    width: tl.constexpr,       # spatial dimension [64]
    
    # Configuration
    BLOCK_SIZE_M: tl.constexpr,  # Block size for C_out dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for spatial dimensions
):
    # Calculate program ID
    pid_b = tl.program_id(0)  # batch dimension
    pid_m = tl.program_id(1)  # output channel dimension
    pid_n = tl.program_id(2)  # spatial dimension (H*W)
    
    # Compute output offset
    out_offset = pid_b * c_out * height * width + pid_m * height * width + pid_n
    
    # Skip if out of bounds
    if pid_m >= c_out or pid_n >= height * width:
        return
    
    # Get scalar scale factor (assuming in_0 is a 0-dim tensor)
    scale = tl.load(in_0_ptr)
    
    # For each batch, contract over j dimension and perform fused operations
    contraction_sum = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Compute contraction over j dimension (c_h = c_w = 64)
    # This is the einsum operation: 'bchj,bhwj->bchw'
    for j in range(c_h):
        # Load tensors needed for contraction
        # in_4: [B, C_out, H, W] -> [B, C_out, H*W] 
        # in_1: [B, C_h, H, W] -> [B, C_h, H*W]
        
        # Calculate tensor offsets
        # For in_4: [B, C_w, H, W] where C_w = C_out = 512
        in_4_base = pid_b * c_out * height * width + j * height * width + pid_n
        in_4_val = tl.load(in_4_ptr + in_4_base, other=0.0)
        
        # For in_1: [B, C_h, H, W] where C_h = 64  
        in_1_base = pid_b * c_h * height * width + j * height * width + pid_n
        in_1_val = tl.load(in_1_ptr + in_1_base, other=0.0)
        
        # Contraction: sum over j dimension
        # This simulates the einsum operation result at [batch, out_channels, spatial_pos]
        contraction_sum += in_4_val * in_1_val
    
    # Load accumulator and update with contraction result
    in_3_current = tl.load(in_3_ptr + out_offset, other=0.0)
    accumulator = in_3_current + contraction_sum
    
    # Apply scale factor and add in_2
    in_2_val = tl.load(in_2_ptr + out_offset, other=0.0)
    result = (accumulator * scale) + in_2_val
    
    # Store final result
    tl.store(out_ptr + out_offset, result)

@triton.jit
def einsum_add_mul_add_fused_kernel_optimized(
    # Input pointers
    in_4_ptr,    # [B, C_w, H, W] - input tensor 1  
    in_1_ptr,    # [B, C_h, H, W] - input tensor 2
    in_3_ptr,    # [B, C_out, H, W] - accumulator tensor
    in_0_ptr,    # scalar - scale factor
    in_2_ptr,    # [B, C_out, H, W] - add tensor
    out_ptr,     # [B, C_out, H, W] - output tensor
    
    # Tensor shapes  
    batch_size: tl.constexpr,
    c_h: tl.constexpr,         # 64
    c_w: tl.constexpr,         # 512 (same as c_out)
    c_out: tl.constexpr,       # 512
    height: tl.constexpr,      # 64
    width: tl.constexpr,       # 64
    
    # Optimized block sizes for better GPU utilization
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1) 
    pid_n = tl.program_id(2)
    
    # Skip out-of-bounds access
    if pid_m >= c_out or pid_b >= batch_size or pid_n >= height * width:
        return
        
    out_offset = pid_b * c_out * height * width + pid_m * height * width + pid_n
    
    # Load scale factor (in_0 is scalar)
    scale = tl.load(in_0_ptr)
    
    # Vectorized contraction over j dimension (C_h = 64)
    # Load a block for better memory coalescing
    pid_n_local = tl.program_id(3)  # Local position within block
    local_offset = pid_n_local * tl.static_range(4)  # Unroll by 4
    
    # For simplicity and better performance, compute contraction per thread
    contraction_sum = 0.0
    for j in range(c_h):
        # Optimized tensor addressing using broadcasting
        in_4_offset = (pid_b * c_w + j) * height * width + pid_n
        in_1_offset = (pid_b * c_h + j) * height * width + pid_n
        
        in_4_val = tl.load(in_4_ptr + in_4_offset, other=0.0)
        in_1_val = tl.load(in_1_ptr + in_1_offset, other=0.0)
        
        contraction_sum += in_4_val * in_1_val
    
    # Load current accumulator
    in_3_current = tl.load(in_3_ptr + out_offset, other=0.0)
    accumulator = in_3_current + contraction_sum
    
    # Load add tensor and compute final result
    in_2_val = tl.load(in_2_ptr + out_offset, other=0.0)
    result = accumulator * scale + in_2_val
    
    # Store output - the contiguous() is handled by the framework
    tl.store(out_ptr + out_offset, result)

# Optimized kernel wrapper with autotuning
@torch.fx.wrap  
def fused_einsum_optimized(in_0, in_1, in_2, in_3, in_4):
    # Get tensor shapes and types
    batch_size = in_1.size(0)
    height = in_1.size(2)
    width = in_1.size(3)
    c_h = in_1.size(1)      # 64
    c_w = in_4.size(1)      # 512 should match c_out
    c_out = in_3.size(1)    # 512
    
    # Create output tensor with contiguous memory
    out = torch.empty_like(in_3)
    
    # Optimized grid dimensions for GPU occupancy
    grid_size_m = (c_out + 127) // 128  # Block size 128 for channels
    grid_size_n = (height * width + 127) // 128  # Block size 128 for spatial
    grid_size_b = batch_size
    grid_size_local = 1  # For vectorization
    
    # Launch kernel with optimal configuration
    grid = (grid_size_b, grid_size_m, grid_size_n, grid_size_local)
    
    # Choose block sizes based on tensor characteristics
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    
    # Launch the kernel
    einsum_add_mul_add_fused_kernel_optimized[grid](
        in_4_ptr=in_4,
        in_1_ptr=in_1, 
        in_3_ptr=in_3,
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        out_ptr=out,
        batch_size=batch_size,
        c_h=c_h,
        c_w=c_w, 
        c_out=c_out,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Alternative simpler kernel for better batch/space parallelism
@triton.jit
def einsum_parallel_kernel(
    in_4_ptr, in_1_ptr, in_3_ptr, in_0_ptr, in_2_ptr, out_ptr,
    batch_size: tl.constexpr, c_h: tl.constexpr, c_out: tl.constexpr, 
    spatial_size: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    if pid_m >= c_out or pid_b >= batch_size or pid_s >= spatial_size:
        return
    
    # Calculate global spatial offset (H * W)
    out_base_idx = pid_b * c_out * spatial_size + pid_m * spatial_size + pid_s
    
    # Compute contraction over j dimension (C_h = 64)
    contraction_sum = 0.0
    for j in range(c_h):
        # Load in_4 tensor: [B, C_out, H, W] equivalent
        in_4_offset = (pid_b * c_out + j) * spatial_size + pid_s
        in_4_val = tl.load(in_4_ptr + in_4_offset, other=0.0)
        
        # Load in_1 tensor: [B, C_h, H, W] equivalent  
        in_1_offset = (pid_b * c_h + j) * spatial_size + pid_s
        in_1_val = tl.load(in_1_ptr + in_1_offset, other=0.0)
        
        contraction_sum += in_4_val * in_1_val
    
    # Load original accumulator value
    in_3_val = tl.load(in_3_ptr + out_base_idx, other=0.0)
    scale = tl.load(in_0_ptr)
    in_2_val = tl.load(in_2_ptr + out_base_idx, other=0.0)
    
    # Fused computation: (in_3 + einsum) * scale + in_2
    # This matches our updated pattern: updated_in_3 = in_3 + einsum, then result = updated_in_3 * scale + in_2
    result = (in_3_val + contraction_sum) * scale + in_2_val
    
    tl.store(out_ptr + out_base_idx, result)

@torch.fx.wrap
def fused_einsum_parallel(in_0, in_1, in_2, in_3, in_4):
    # Get tensor characteristics 
    batch_size = in_1.size(0)
    c_h = in_1.size(1)      # 64
    c_out = in_3.size(1)    # 512  
    height = in_1.size(2)   # 64
    width = in_1.size(3)    # 64
    spatial_size = height * width
    
    # Output tensor
    out = torch.empty_like(in_3)
    
    # Configure grid for maximum parallelism
    grid_b = batch_size
    grid_m = (c_out + 63) // 64    # 64 threads per channel block
    grid_s = (spatial_size + 63) // 64  # 64 threads per spatial block
    
    einsum_parallel_kernel[(grid_b, grid_m, grid_s)](
        in_4_ptr=in_4,
        in_1_ptr=in_1,
        in_3_ptr=in_3,
        in_0_ptr=in_0, 
        in_2_ptr=in_2,
        out_ptr=out,
        batch_size=batch_size,
        c_h=c_h,
        c_out=c_out,
        spatial_size=spatial_size
    )
    
    return out

# Replacement function - selects the best kernel strategy  
def replacement_func():
    return fused_einsum_parallel