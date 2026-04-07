import torch
import triton
import triton.language as tl

# Pattern matching function - match BOTH operations together
def pattern(in_0, in_1):
    """Match both scalar multiplication and transpose operations"""
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.transpose(-2, -1)
    return (tmp_0, tmp_1)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel
@triton.jit
def fused_kernel_scalarmul_transpose(
    in_1_ptr,          # Pointer to input tensor 1 (for scalar multiplication)
    in_0_ptr,          # Pointer to input tensor 0 (for transpose)  
    out_0_ptr,         # Output for scalar multiplication result
    out_1_ptr,         # Output for transpose result
    
    # Tensor shapes and strides
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    
    # Strides for tensor 0 (to be transposed)
    stride_B0: tl.constexpr, stride_C0: tl.constexpr, stride_H0: tl.constexpr, stride_W0: tl.constexpr,
    
    # Strides for tensor 1 (scalar multiplication)
    stride_B1: tl.constexpr, stride_C1: tl.constexpr, stride_H1: tl.constexpr, stride_W1: tl.constexpr,
    
    scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for scalar multiplication + transpose operations"""
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (B * C * H * W)
    
    if tl.all(~mask):
        return
        
    # ======== Scalar Multiplication Section ========
    # Convert linear index to coordinates for tensor 1 [B, C, H, W]
    idx_1 = offsets
    b1 = idx_1 // stride_B1
    remainder = idx_1 % stride_B1
    c1 = remainder // stride_C1
    remainder = remainder % stride_C1
    h1 = remainder // stride_H1
    w1 = remainder % stride_H1
    
    # Calculate linear index and load from tensor 1
    input_idx_1 = b1 * stride_B1 + c1 * stride_C1 + h1 * stride_H1 + w1 * stride_W1
    in_1_val = tl.load(in_1_ptr + input_idx_1, mask=mask)
    
    # Perform scalar multiplication
    out_0_val = in_1_val * scalar
    
    # Store scalar multiplication result (output shape same as input [B, C, H, W])
    linear_idx_out_0 = offsets  # Same input/output layout for scalar multiply
    tl.store(out_0_ptr + linear_idx_out_0, out_0_val, mask=mask)
    
    # ======== Transpose Section ========
    # Convert linear index in transposed output to coordinates [B, C, W, H]
    idx_trans = offsets
    b_trans = idx_trans // (C * W * H)
    remainder = idx_trans % (C * W * H)
    c_trans = remainder // (W * H)
    remainder = remainder % (W * H)
    w_trans = remainder // H
    h_trans = remainder % H
    
    # Calculate original coordinates [B, C, H, W] -> transpose to [B, C, W, H]
    input_idx_0 = b_trans * stride_B0 + c_trans * stride_C0 + h_trans * stride_H0 + w_trans * stride_W0
    
    # Load from original position and store to transposed position
    in_0_val = tl.load(in_0_ptr + input_idx_0, mask=mask)
    tl.store(out_1_ptr + offsets, in_0_val, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_fused_operations(in_0, in_1):
    """Wrapper function for fused scalar multiplication + transpose"""
    # Get tensor properties
    B, C, H, W = in_0.shape
    scalar_const = 0.1767766952966369
    
    # Calculate strides (assuming contiguous tensors for now)
    stride_B0 = C * H * W
    stride_C0 = H * W
    stride_H0 = W
    stride_W0 = 1
    
    stride_B1 = C * H * W
    stride_C1 = H * W
    stride_H1 = W
    stride_W1 = 1
    
    # Total elements
    total_elements = B * C * H * W
    
    # Optimize block size - larger blocks for better occupancy
    BLOCK_SIZE = 2048  # Larger block size to reduce kernel launch overhead
    
    # Calculate number of programs
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    out_0 = torch.empty_like(in_1)  # Same shape as in_1
    out_1 = torch.empty((B, C, W, H), dtype=in_0.dtype, device=in_0.device)
    
    # Launch fused kernel
    fused_kernel_scalarmul_transpose[(num_programs,)](
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        out_0_ptr=out_0,
        out_1_ptr=out_1,
        B=B, C=C, H=H, W=W,
        stride_B0=stride_B0, stride_C0=stride_C0, stride_H0=stride_H0, stride_W0=stride_W0,
        stride_B1=stride_B1, stride_C1=stride_C1, stride_H1=stride_H1, stride_W1=stride_W1,
        scalar=scalar_const,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_0, out_1)

# Replacement function (returns function reference)
def replacement_func():
    return triton_fused_operations