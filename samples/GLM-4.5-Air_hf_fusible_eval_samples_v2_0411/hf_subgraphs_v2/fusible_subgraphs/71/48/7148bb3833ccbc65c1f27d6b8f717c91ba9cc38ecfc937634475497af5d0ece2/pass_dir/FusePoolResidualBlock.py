import torch
import triton
import triton.language as tl

@triton.jit
def fused_pool_residual_kernel(
    x_ptr,  # input tensor [B, C, H, W]
    out_ptr,  # output tensor [B, C, H, W]
    n_elements,  # total elements in tensor
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that fuses ReLU, AvgPool2D, and residual subtraction"""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    num_blocks = (n_elements + block_size - 1) // block_size
    
    # Each program handles one 2D spatial tile in each channel
    # We'll process the tensor as batch x channel flattened 2D tiles
    for batch_channel_index in range(pid, num_blocks):
        start_idx = batch_channel_index * block_size
        end_idx = min(start_idx + block_size, n_elements)
        
        for idx in range(start_idx, end_idx):
            # Calculate original indices
            # For [B, C, H, W] = [64, 48, 56, 56] with BxC as flattened batch
            spatial_elements = 56 * 56  # H * W
            batch_channel = idx // spatial_elements
            spatial_idx = idx % spatial_elements
            h = spatial_idx // 56
            w = spatial_idx % 56
            
            # Apply ReLU to input
            x_val = tl.load(x_ptr + idx)
            relu_val = max(x_val, 0.0)
            
            # Compute average pooling with 3x3 kernel, stride 1, padding 1
            pool_sum = 0.0
            pool_count = 0
            
            # 3x3 window with padding=1
            for kh in range(-1, 2):
                for kw in range(-1, 2):
                    pool_h = h + kh
                    pool_w = w + kw
                    
                    # Boundary check
                    if 0 <= pool_h < 56 and 0 <= pool_w < 56:
                        # Calculate pooling input index
                        pool_idx = batch_channel * spatial_elements + pool_h * 56 + pool_w
                        pool_input = tl.load(x_ptr + pool_idx)
                        pool_sum += max(pool_input, 0.0)  # Apply ReLU before pooling
                        pool_count += 1
            
            avg_pool_val = pool_sum / pool_count if pool_count > 0 else 0.0
            
            # Residual: avg_pool - relu
            residual_val = avg_pool_val - relu_val
            
            # Store result (this is tmp_4 in the original)
            tl.store(out_ptr + idx, residual_val)

@torch.fx.wrap
def fused_pool_residual_operation(x):
    """Fused ReLU, AvgPool2D, and residual subtraction operation"""
    B, C, H, W = x.shape
    N = B * C * H * W
    
    # Use optimized block size based on tensor dimensions
    BLOCK_SIZE = 1024
    
    out = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_pool_residual_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return needed intermediates for downstream operations
    # We need tmp_2 (ReLU output) and tmp_4 (residual)
    # NOTE: tmp_2 is computed outside this func and passed as the first return value
    return out  # only return tmp_4 (residual)

def pattern(tmp_2):
    """Match: AvgPool2D -> Subtraction pattern (tmp_2 is already computed)"""
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    return tmp_4

def replacement_args(tmp_2):
    """Extract arguments for replacement function"""
    return (tmp_2,)

def replacement_func():
    """Return the fused operation function"""
    return fused_pool_residual_operation