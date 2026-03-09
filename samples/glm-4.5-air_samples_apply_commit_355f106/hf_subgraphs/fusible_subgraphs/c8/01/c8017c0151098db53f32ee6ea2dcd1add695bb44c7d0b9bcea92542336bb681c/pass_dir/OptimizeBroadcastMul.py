import torch
import triton
import triton.language as tl

def pattern(a, b):
    # Pattern: Element-wise multiplication with broadcast (a * b)
    result = a * b
    return result

def replacement_args(a, b):
    return (a, b)

@triton.jit
def optimized_broadcast_mul_kernel(
    a_ptr, b_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for element-wise multiplication with broadcast"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H * W
    
    # Calculate 4D indices
    linear_idx = offsets
    w = linear_idx % W
    linear_idx = linear_idx // W
    h = linear_idx % H
    linear_idx = linear_idx // H
    c = linear_idx % C
    n = linear_idx // C
    
    # Load input A (full 4D tensor)
    a_offset = n * C * H * W + c * H * W + h * W + w
    a_val = tl.load(a_ptr + a_offset, mask=mask)
    
    # Load input B (may be broadcasted)
    # Check if B has spatial dimensions > 1
    if H > 1 or W > 1:
        # B has full 4D shape - load directly
        b_offset = n * C * H * W + c * H * W + h * W + w
        b_val = tl.load(b_ptr + b_offset, mask=mask)
    else:
        # B is broadcasted - load from batch and channel only
        b_2d_offset = n * C + c
        b_val = tl.load(b_ptr + b_2d_offset)
    
    # Perform multiplication
    out_val = a_val * b_val
    
    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)

@triton.jit
def optimized_broadcast_mul_2d_kernel(
    a_ptr, b_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for when B is fully broadcasted [N, C, 1, 1] -> [N, C, H, W]"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C * H * W
    
    # Calculate 4D indices
    linear_idx = offsets
    w = linear_idx % W
    linear_idx = linear_idx // W
    h = linear_idx % H
    linear_idx = linear_idx // H
    c = linear_idx % C
    n = linear_idx // C
    
    # Load input A (full 4D tensor)
    a_offset = n * C * H * W + c * H * W + h * W + w
    a_val = tl.load(a_ptr + a_offset, mask=mask)
    
    # Load input B (broadcasted from [N, C, 1, 1])
    b_offset = n * C + c  # Only batch and channel matter
    b_val = tl.load(b_ptr + b_offset)
    
    # Perform multiplication
    out_val = a_val * b_val
    
    # Store result
    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def optimized_broadcast_mul(a, b):
    """Optimized element-wise multiplication with broadcast for 4D tensors"""
    if a.numel() == 0 or b.numel() == 0:
        return torch.empty_like(a)
    
    Na, Ca, Ha, Wa = a.shape
    Nb, Cb, Hb, Wb = b.shape
    
    # Output shape follows broadcasting rules
    N_out = max(Na, Nb)
    C_out = max(Ca, Cb)
    H_out = max(Ha, Hb)
    W_out = max(Wa, Wb)
    
    out_shape = (N_out, C_out, H_out, W_out)
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    num_programs = (N_out * C_out * H_out * W_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Choose kernel based on broadcast patterns
    # Common case: b is [N, C, 1, 1] being broadcasted to [N, C, H, W]
    if Hb == 1 and Wb == 1 and Cb == C_out and Nb == N_out:
        optimized_broadcast_mul_2d_kernel[(num_programs,)](
            a_ptr=a,
            b_ptr=b,
            out_ptr=out,
            N=N_out, C=C_out, H=H_out, W=W_out,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        optimized_broadcast_mul_kernel[(num_programs,)](
            a_ptr=a,
            b_ptr=b,
            out_ptr=out,
            N=N_out, C=C_out, H=H_out, W=W_out,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    return optimized_broadcast_mul