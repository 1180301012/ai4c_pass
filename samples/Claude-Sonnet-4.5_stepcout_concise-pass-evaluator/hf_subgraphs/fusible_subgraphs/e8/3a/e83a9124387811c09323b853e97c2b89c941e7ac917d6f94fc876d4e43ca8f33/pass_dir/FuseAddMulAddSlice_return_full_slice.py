import torch
import triton
import triton.language as tl

# Pattern for fused operation
def pattern(in_0, in_1, in_2, in_3):
    """Match the pattern: (in_3 + in_2) * in_1 + in_0"""
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_add_mul_add_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    B,
    S,
    D,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Fused kernel for (in_3 + in_2) * in_1 + in_0 with broadcasting"""
    # Each program handles one (batch, seq) position
    pid_bs = tl.program_id(0)
    
    # Calculate base offset in the 3D tensor [B, S, D]
    base_offset = pid_bs * D
    
    # Process D elements in one shot
    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D
    
    # Calculate offsets
    offsets_3d = base_offset + d_offsets
    
    # Load from 3D tensors (coalesced)
    in_2 = tl.load(in_2_ptr + offsets_3d, mask=d_mask, other=0.0)
    in_3 = tl.load(in_3_ptr + offsets_3d, mask=d_mask, other=0.0)
    
    # Load from 1D tensors (broadcast - also coalesced)
    in_0 = tl.load(in_0_ptr + d_offsets, mask=d_mask, other=0.0)
    in_1 = tl.load(in_1_ptr + d_offsets, mask=d_mask, other=0.0)
    
    # Compute: (in_3 + in_2) * in_1 + in_0
    result = (in_3 + in_2) * in_1 + in_0
    
    # Store result (coalesced)
    tl.store(out_ptr + offsets_3d, result, mask=d_mask)

@torch.fx.wrap
def fused_add_mul_add_wrapper(in_0, in_1, in_2, in_3):
    """
    Compute: tmp_4 = (in_3 + in_2) * in_1 + in_0
    
    in_0, in_1: shape [D] - bias and scale
    in_2, in_3: shape [B, S, D] - input tensors
    """
    # Get shapes
    B, S, D = in_2.shape
    total_elements = B * S * D
    
    # Only use Triton for very large tensors where we see actual speedup
    # Threshold: ~4M elements based on empirical results
    if total_elements < 4000000:
        # Use native PyTorch operations for small/medium tensors
        return (in_3 + in_2) * in_1 + in_0
    
    # Allocate output
    tmp_4 = torch.empty_like(in_2)
    
    # Choose block size dynamically based on D
    if D <= 64:
        BLOCK_SIZE_D = 64
    elif D <= 128:
        BLOCK_SIZE_D = 128
    elif D <= 256:
        BLOCK_SIZE_D = 256
    elif D <= 512:
        BLOCK_SIZE_D = 512
    else:
        BLOCK_SIZE_D = 1024
    
    # Launch fused kernel - one thread block per (B, S) position
    grid = (B * S,)
    fused_add_mul_add_kernel[grid](
        in_0, in_1, in_2, in_3, tmp_4,
        B, S, D,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    
    return tmp_4

def replacement_func():
    return fused_add_mul_add_wrapper