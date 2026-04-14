import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # The complete computation pattern from model.py
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    return (tmp_6, tmp_4)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def complete_fusion_kernel(
    in_0_ptr,
    in_1_ptr, 
    in_2_ptr,
    out_1_ptr,  # tmp_6 (permuted result)
    out_2_ptr,  # tmp_4 (direct result)
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
):
    """Complete end-to-end fusion kernel for the entire computation"""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Block size for C dimension (128 elements)
    block_start_c = pid_c * 128
    mask_c = block_start_c + tl.arange(0, 128) < C
    
    # Compute base addresses for each batch element and height
    in_2_base = in_2_ptr + pid_b * (H * 1 * C) + pid_h * (1 * C)
    
    # Load in_2 data: [B, H, 1, C]
    in_2_data = tl.load(in_2_base + block_start_c, mask=mask_c, other=0.0)
    
    # Load in_0 data: [2, C] - use appropriate slices for each W dimension
    # For W=0 slice, use in_0[0:1] = [0:1, :]
    # For W=1 slice, use in_0[1:2] = [1:2, :]
    in_0_slice0_base = in_0_ptr + 0 * C  # First row of in_0
    in_0_slice1_base = in_0_ptr + 1 * C  # Second row of in_0
    
    in_0_slice0 = tl.load(in_0_slice0_base + block_start_c, mask=mask_c, other=0.0)
    in_0_slice1 = tl.load(in_0_slice1_base + block_start_c, mask=mask_c, other=0.0)
    
    # Load in_1 data: [2, C] - effective shape after broadcasting
    # in_1 is [1,1,2,C], we extract the [2,C] part
    in_1_slice0_base = in_1_ptr + 0 * C  # First slice
    in_1_slice1_base = in_1_ptr + 1 * C  # Second slice
    
    in_1_slice0 = tl.load(in_1_slice0_base + block_start_c, mask=mask_c, other=0.0)
    in_1_slice1 = tl.load(in_1_slice1_base + block_start_c, mask=mask_c, other=0.0)
    
    # Perform fused computation: (in_2 * in_1 + in_0) for both W slices
    # Result should be [B, H, 2, C] but we store directly to final outputs
    
    # For W=0: in_2 * in_1[0] + in_0[0]
    result_slice0 = in_2_data * in_1_slice0 + in_0_slice0
    
    # For W=1: in_2 * in_1[1] + in_0[1] (in_2 remains same due to broadcasting)
    result_slice1 = in_2_data * in_1_slice1 + in_0_slice1
    
    # Store tmp_4: result_slice0 as [B, H, 1, C] -> stored as [B, H, C]
    out_2_addr = out_2_ptr + pid_b * (H * C) + pid_h * C + block_start_c + tl.arange(0, 128)
    tl.store(out_2_addr, result_slice0, mask=mask_c)
    
    # Store tmp_6: result_slice1 as [B, 1, H, C] (permuted)
    # Original: [b, h, 1, c], target: [b, 1, h, c]
    # We need to store this in the output tensor at position [b, h, c]
    permuted_addr = out_1_ptr + pid_b * (H * C) + pid_h * C + block_start_c + tl.arange(0, 128)
    tl.store(permuted_addr, result_slice1, mask=mask_c)

@torch.fx.wrap
def complete_fusion_compute(in_0, in_1, in_2):
    # Get input shapes
    B = in_2.shape[0]  # Batch size from in_2
    H = in_2.shape[1]  # 17 from weight_meta
    C = in_2.shape[3]  # 128 from weight_meta
    
    # Create output tensors
    tmp_4 = torch.empty((B, H, 1, C), dtype=in_0.dtype, device=in_0.device)  # [B, H, 1, C]
    tmp_6 = torch.empty((B, 1, H, C), dtype=in_0.dtype, device=in_0.device)  # [B, 1, H, C] after permute
    
    # Launch complete fusion kernel
    grid = lambda meta: (meta['B'], meta['H'], (meta['C'] + 127) // 128)
    
    complete_fusion_kernel[grid](
        in_0, in_1, in_2, tmp_6, tmp_4,
        B=B, H=H, C=C
    )
    
    return (tmp_6, tmp_4)

def replacement_func():
    return complete_fusion_compute