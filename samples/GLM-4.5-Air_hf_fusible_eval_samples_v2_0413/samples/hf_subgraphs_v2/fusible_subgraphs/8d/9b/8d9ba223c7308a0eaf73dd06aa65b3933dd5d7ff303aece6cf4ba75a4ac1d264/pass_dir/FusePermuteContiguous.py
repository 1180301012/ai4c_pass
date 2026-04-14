import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Match view(64, 64, -1) -> permute(2, 0, 1) -> contiguous() sequence"""
    # Based on the computation pattern:
    # tmp_6 = input_tensor.view(64, 64, -1)
    # tmp_7 = tmp_6.permute(2, 0, 1)  
    # tmp_8 = tmp_7.contiguous()
    # We match the permute(2, 0, 1) + contiguous sequence
    tmp_7 = input_tensor.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    return tmp_8

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def permute_contiguous_kernel(
    input_ptr,
    output_ptr, 
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that fuses permute(2, 0, 1) with contiguous copy"""
    # This kernel handles input of shape (H, W, C) -> output of shape (C, H, W)
    # Using direct memory access pattern optimization
    
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (H * W * C)
    
    k = offsets
    
    # Simplified calculation: for (H, W, C) -> (C, H, W) permutation
    # Output index: [C, H, W] -> k = c * H * W + h * W + w
    # Input index: [H, W, C] -> offset = h * W * C + w * C + c
    # So we need to map k to input_offset
    c = k // (H * W)
    remainder = k % (H * W)
    h = remainder // W
    w = remainder % W
    
    # Calculate input offset using the original memory layout
    input_offset = h * W * C + w * C + c
    
    # Direct memory transfer with permuted layout
    val = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + k, val, mask=mask)

@torch.fx.wrap
def fused_permute_contiguous(a):
    """Wrapper function for fused permute(2, 0, 1) + contiguous operation"""
    if a.dim() != 3:
        # Fallback for non-3D tensors
        return a.permute(2, 0, 1).contiguous()
    
    # Input is (H, W, C), output is (C, H, W) due to permute(2, 0, 1)
    H, W, C = a.shape[0], a.shape[1], a.shape[2]
    out = torch.empty((C, H, W), dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(H * W * C, meta['BLOCK_SIZE']),)
    
    permute_contiguous_kernel[grid](
        a,
        out,
        H,
        W, 
        C,
        1024,  # BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_permute_contiguous