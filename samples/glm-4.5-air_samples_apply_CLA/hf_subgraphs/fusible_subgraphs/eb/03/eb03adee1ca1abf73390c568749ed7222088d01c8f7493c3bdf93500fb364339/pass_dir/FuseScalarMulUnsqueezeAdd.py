import torch
import triton
import triton.language as tl

def pattern(x, y, scale):
    # Pattern matches: scaled_y = y * scale, unsqueezed_x = x.unsqueeze(2), result = scaled_y + unsqueezed_x
    scaled_y = y * scale
    unsqueezed_x = x.unsqueeze(2)
    result = scaled_y + unsqueezed_x
    return result

def replacement_args(x, y, scale):
    return (x, y, scale)

@triton.jit
def fused_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_shape0,
    x_shape1, 
    x_shape2,
    x_shape3,
    y_shape0,
    y_shape1,
    y_shape2,
    y_shape3,
    y_shape4,
    scale_val,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Calculate offsets for x [1, 361, 1, 49, 49] -> [361, 1, 49, 49] after removing batch
    # and y [1, 361, 3, 49, 49] -> [361, 3, 49, 49] after removing batch
    
    # We'll process the batch dimension differently - it's size 1
    # Process the 361 heads dimension
    m_offset = pid_m
    # For x: [361, 1, 49, 49], for y: [361, 3, 49, 49]
    n_offset_x = 0  # x has only 1 element in this dimension
    n_offset_y = pid_n  # y has 3 elements in this dimension
    k_offset = pid_k
    
    mask_n_y = n_offset_y < 3  # y has dimension size 3
    mask_k = k_offset < 49        # both have dimension size 49
    
    # Use separate if statements to avoid chained boolean operators
    if not (m_offset < 361):
        return
    if not mask_n_y:
        return
    if not mask_k:
        return
    
    # Calculate global offsets
    # x_ptr: [361, 1, 49, 49] -> m=361, n=1, k0=49, k1=49
    x_offset = (m_offset * 1 * 49 * 49) + (n_offset_x * 49 * 49) + (k_offset * 49)
    # y_ptr: [361, 3, 49, 49] -> m=361, n=3, k0=49, k1=49  
    y_offset = (m_offset * 3 * 49 * 49) + (n_offset_y * 49 * 49) + (k_offset * 49)
    
    # Load elements with broadcasting semantics
    # x is [361, 1, 49, 49], y is [361, 3, 49, 49]
    # We need to broadcast x to match y's shape [361, 3, 49, 49]
    x_val = tl.load(x_ptr + x_offset, mask=mask_k, other=0.0)
    y_val = tl.load(y_ptr + y_offset, mask=mask_n_y & mask_k, other=0.0)
    
    # Perform fused computation: result = (y * scale) + x (with broadcasting)
    out_val = (y_val * scale_val) + x_val
    
    # Store result
    out_offset = y_offset  # result has same shape as y
    tl.store(out_ptr + out_offset, out_val, mask=mask_n_y & mask_k)

# Define the decorated function at module level
@torch.fx.wrap
def fused_scalar_mul_unsqueeze_add(x, y, scale):
    # Handle batch dimension (size 1)
    # Remove batch dimension for processing: [1, 361, 49, 49] -> [361, 49, 49]
    x_no_batch = x.squeeze(0)  # [361, 49, 49]
    y_no_batch = y.squeeze(0)  # [361, 3, 49, 49]
    
    # Unsqueeze x at dimension 2 to match broadcasting pattern
    x_unsqueeze = x_no_batch.unsqueeze(1)  # [361, 1, 49, 49]
    
    # Calculate output shape
    out_shape = [361, 3, 49, 49]
    
    # Output tensor
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Grid configuration
    M = 361  # heads
    N = 3    # the dimension we broadcast over
    K = 49 * 49  # spatial dimensions flattened
    
    BLOCK_SIZE_K = 64  # reduce K dimension
    num_programs_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Launch kernel with 3D grid
    fused_kernel[(M, N, num_programs_k)](
        x_ptr=x_unsqueeze,
        y_ptr=y_no_batch, 
        out_ptr=out,
        x_shape0=361, x_shape1=1, x_shape2=49, x_shape3=49,
        y_shape0=361, y_shape1=3, y_shape2=49, y_shape3=49,
        y_shape4=49,
        scale_val=scale,
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=1, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Add batch dimension back
    return out.unsqueeze(0)

def replacement_func():
    return fused_scalar_mul_unsqueeze_add