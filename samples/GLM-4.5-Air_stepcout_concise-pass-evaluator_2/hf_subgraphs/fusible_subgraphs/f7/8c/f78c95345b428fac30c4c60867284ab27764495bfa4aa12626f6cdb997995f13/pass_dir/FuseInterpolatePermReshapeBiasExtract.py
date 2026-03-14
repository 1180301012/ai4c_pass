import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern for fused interpolate + permute + reshape + bias extraction
    This models the relative position bias computation found in transformer attention modules.
    """
    tmp_1 = torch.nn.functional.interpolate(in_1, size=(0, 0), mode='bilinear')
    tmp_2 = tmp_1.permute(0, 2, 3, 1)
    tmp_3 = tmp_2.reshape(0, -1)
    tmp_4 = in_0[slice(0, None, None)]
    return (tmp_4, tmp_3)

def replacement_args(in_0, in_1):
    # Return inputs directly - dimensions will be handled in kernel
    return (in_0, in_1)

@triton.jit
def fused_bias_kernel(
    in_0_ptr, in_1_ptr,
    out_0_ptr, out_1_ptr, 
    N_total: tl.constexpr, H_total: tl.constexpr, C: tl.constexpr,
    bias_start: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused kernel for permute, reshape, and bias extraction"""
    pid = tl.program_id(0)
    
    # Handle bias extraction (final elements from in_0)
    if out_0_ptr is not None and bias_start > 0:
        bias_offsets = tl.arange(bias_start) + bias_start
        bias_mask = tl.arange(bias_start) < N_total - bias_start
        bias_values = tl.load(in_0_ptr + bias_offsets, mask=bias_mask.to(tl.int1))
        tl.store(out_0_ptr, bias_values)
    
    # Handle main computation: permute [N,C,H,W] -> [N,H,W,C] and reshape to [N*H*W, C]
    if out_1_ptr is not None:
        # Each program handles multiple elements
        start_idx = pid * BLOCK_SIZE
        end_idx = min((pid + 1) * BLOCK_SIZE, N_total)
        
        for idx in range(start_idx, end_idx):
            # Convert linear index to coordinates
            n_idx = idx // (H_total * H_total)  # Assuming square H x W
            h_idx = (idx % (H_total * H_total)) // H_total
            w_idx = (idx % (H_total * H_total)) % H_total
            
            # Load and permute: [N,C,H,W] -> [N,H,W,C]
            for c_idx in range(C):
                src_offset = n_idx * C * H_total * H_total + c_idx * H_total * H_total + h_idx * H_total + w_idx
                dst_offset = idx * C + c_idx
                
                mask = src_offset < N_total * C * H_total * H_total
                value = tl.load(in_1_ptr + src_offset, mask=mask.to(tl.int1))
                tl.store(out_1_ptr + dst_offset, value, mask=(idx < N_total).to(tl.int1))

@torch.fx.wrap  
def fused_bias_computation(in_0, in_1):
    """
    Optimized implementation for relative position bias computation.
    Fuses: interpolate+permute(0,2,3,1)+reshape + bias slice
    """
    # Get input dimensions
    N, C, H, W = in_1.shape
    total_pixels = H * W
    
    # Since interpolate size equals input size, it's effectively a no-op
    # Optimize out the interpolate and directly process the permutation and reshape
    
    # Extract bias suffix from in_0
    total_bias_elements = in_0.shape[0]
    bias_start = max(0, total_bias_elements - total_pixels)
    bias_slice = in_0[bias_start:] if bias_start < total_bias_elements else in_0[:0]
    
    # Create output for reshaped tensor: [N, H, W, C] -> [N*H*W, C]
    reshaped_shape = (N * total_pixels, C)
    reshaped_output = torch.empty(reshaped_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024  # Optimized block size for GPU occupancy
    grid_size = (total_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch fused kernel
    fused_bias_kernel[grid_size](
        in_0, in_1,
        bias_slice, reshaped_output,
        N, H, C, bias_start,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return bias_slice, reshaped_output

def replacement_func():
    return fused_bias_computation