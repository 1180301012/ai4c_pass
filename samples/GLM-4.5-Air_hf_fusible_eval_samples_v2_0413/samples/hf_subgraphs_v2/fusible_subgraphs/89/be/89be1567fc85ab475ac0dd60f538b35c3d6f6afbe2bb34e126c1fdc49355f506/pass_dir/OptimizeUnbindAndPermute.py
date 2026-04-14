import torch
import triton
import triton.language as tl

def pattern(tmp_2):
    # The exact computation pattern for unbind + permute
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    return tmp_6, tmp_4

def replacement_args(tmp_2):
    return (tmp_2,)

@triton.jit
def optimized_unbind_permute_kernel(
    in_ptr,
    out_1_ptr,  # tmp_6 (permuted)
    out_2_ptr,  # tmp_4 (direct)
    B: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
):
    """Optimized kernel that directly extracts and permutes without intermediate tensors"""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Grid: [B, C] with 128 elements per program
    block_start_c = pid_c * 128
    mask_c = block_start_c + tl.arange(0, 128) < C
    
    # Input address: [B, H, W, C] 
    in_base_addr = in_ptr + pid_b * (H * W * C)
    
    # Extract both tensors directly from input without unbind
    # tensor_1: [B, H, 1, C] - first slice along W dimension
    tensor_1_addr = in_base_addr + block_start_c
    
    # tensor_2: [B, H, 1, C] - second slice along W dimension  
    tensor_2_addr = in_base_addr + (H * 1 * C) + block_start_c
    
    # Load both tensors
    tensor_1 = tl.load(tensor_1_addr, mask=mask_c, other=0.0)
    tensor_2 = tl.load(tensor_2_addr, mask=mask_c, other=0.0)
    
    # Directly compute permuted tensor_2: [B, H, 1, C] -> [B, 1, H, C]
    # Permutation: (0, 2, 1) means we reorder dimensions 0->0, 2->1, 1->2
    # For each element [b, h, 1, c] -> [b, 1, h, c]
    # Since H dimension is 17 and we have 128 elements per thread, we need to handle this carefully
    
    # For the permuted output, we need to reorder memory layout
    # Original: [b, h, 1, c] -> [B, H, C] contiguous in memory
    # Permuted: [b, 1, h, c] -> [B, H, C] but with stride changes
    
    # Store unpermuted tensor (tmp_4): [B, H, 1, C] as [B, H, C] 
    out_2_addr = out_2_ptr + pid_b * (H * C) + pid_c * 128 + tl.arange(0, 128)
    tl.store(out_2_addr, tensor_1, mask=(pid_c * 128 + tl.arange(0, 128)) < C)
    
    # Store permuted tensor (tmp_6): [B, 1, H, C] 
    # For each b and c, store H elements in correct order
    h_indices = tl.arange(0, H)
    c_offset = 128 * pid_c
    
    for h in range(H):
        # For each h, store the corresponding element in permuted layout
        permuted_addr = out_1_ptr + pid_b * (H * C) + h * C + c_offset + tl.arange(0, 128)
        mask = (c_offset + tl.arange(0, 128)) < C
        # Load original element [b, h, 0, c] and store at [b, h, c] in permuted layout
        orig_addr = in_base_addr + h * (1 * C) + c_offset + tl.arange(0, 128)
        orig_val = tl.load(orig_addr, mask=mask, other=0.0)
        tl.store(permuted_addr, orig_val, mask=mask)

@torch.fx.wrap  
def optimized_unbind_permute(tmp_2):
    B, H, W, C = tmp_2.shape
    assert W == 2, "This optimization assumes W dimension is 2 for unbind"
    
    # Create output tensors
    tmp_4 = torch.empty((B, H, 1, C), dtype=tmp_2.dtype, device=tmp_2.device)  # [B, H, 1, C]
    tmp_6 = torch.empty((B, 1, H, C), dtype=tmp_2.dtype, device=tmp_2.device)  # [B, 1, H, C] after permute
    
    # Launch kernel
    grid = lambda meta: (meta['B'], (meta['C'] + 127) // 128)
    
    optimized_unbind_permute_kernel[grid](
        tmp_2, tmp_6, tmp_4,
        B=B, H=H, W=W, C=C
    )
    
    return tmp_6, tmp_4

def replacement_func():
    return optimized_unbind_permute