import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match the pattern: sum(dim=1) followed by adaptive_avg_pool2d(x, 1)
    This pattern computes a global reduction: sum over channel dim, then mean over spatial dims.
    """
    tmp_0 = in_0.sum(dim=1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_sum_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    stride_b: tl.constexpr,
    stride_c: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """
    Optimized kernel: Each thread processes one (batch, depth) output.
    Uses vectorized loads for better memory throughput.
    
    Input shape: [B, C, D, H, W]
    Output shape: [B, D, 1, 1]
    """
    # Get program id for batch and depth
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    # Calculate output pointer offset
    output_offset = pid_b * D + pid_d
    output_ptr_at = output_ptr + output_offset
    
    # Total number of spatial elements to reduce
    num_spatial = H * W
    
    # Initialize accumulator
    sum_val = 0.0
    
    # Process H*W elements in blocks
    # For each h,w position, sum over all c
    for block_start in range(0, num_spatial, BLOCK_SIZE_HW):
        # Create offsets for H*W dimension
        hw_offsets = block_start + tl.arange(0, BLOCK_SIZE_HW)
        h_idx = hw_offsets % H
        w_idx = hw_offsets // H
        
        # Create mask for valid elements
        mask = hw_offsets < num_spatial
        
        # For each h,w position, we need to sum over all c
        # Let's process channel dimension inside the loop
        c_sum = 0.0
        for c_idx in range(0, C):
            # Calculate input offset for this c,h,w
            input_offset = (
                pid_b * stride_b + 
                c_idx * stride_c + 
                pid_d * stride_d + 
                h_idx * stride_h + 
                w_idx * stride_w
            )
            
            # Load and accumulate
            vals = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
            c_sum += tl.sum(vals, axis=0)
        
        sum_val += c_sum
    
    # Compute mean over spatial dimensions (H*W)
    mean_val = sum_val / (H * W)
    
    # Store result
    tl.store(output_ptr_at, mean_val)


def fused_sum_avg_pool2d(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Fused kernel that performs sum over dim=1 followed by adaptive_avg_pool2d with output_size=1.
    
    Input shape: [B, C, D, H, W]
    Output shape: [B, D, 1, 1]
    """
    B, C, D, H, W = input_tensor.shape
    
    # Output tensor: [B, D, 1, 1]
    output = torch.zeros((B, D, 1, 1), dtype=torch.float32, device=input_tensor.device)
    
    # Block size for spatial dimensions (H*W)
    BLOCK_SIZE_HW = 1024
    
    # Grid: (B, D)
    grid = (B, D)
    
    fused_sum_avg_pool2d_kernel[grid](
        input_tensor,
        output,
        B, C, D, H, W,
        input_tensor.stride(0),
        input_tensor.stride(1),
        input_tensor.stride(2),
        input_tensor.stride(3),
        input_tensor.stride(4),
        BLOCK_SIZE_HW,
    )
    
    return output


@torch.fx.wrap
def fused_kernel_wrapper(in_0):
    return fused_sum_avg_pool2d(in_0)


def replacement_func():
    return fused_kernel_wrapper