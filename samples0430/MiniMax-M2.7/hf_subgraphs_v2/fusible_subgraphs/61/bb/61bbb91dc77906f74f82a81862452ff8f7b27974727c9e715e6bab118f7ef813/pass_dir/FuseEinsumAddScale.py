import torch
import triton
import triton.language as tl

@triton.jit
def fused_ccnet_kernel(
    # Pointers
    in_1_ptr, in_2_ptr, in_3_ptr, in_4_ptr, out_ptr,
    # Shapes
    B, C, H, W, J,
    # Scalar
    scale,
    # Strides
    in_1_sB, in_1_sH, in_1_sW, in_1_sJ,
    in_2_sB, in_2_sC, in_2_sH, in_2_sW,
    in_3_sB, in_3_sC, in_3_sH, in_3_sJ,
    in_4_sB, in_4_sC, in_4_sH, in_4_sJ,
    out_sB, out_sC, out_sH, out_sW,
    # Block sizes
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    J_BLOCK: tl.constexpr,
):
    # Get program IDs
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Calculate offsets for output [B, C, H, W]
    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    off_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    off_w = tl.arange(0, BLOCK_SIZE_W)
    
    # Mask for bounds checking
    mask_b = off_b < B
    mask_c = off_c < C
    mask_h = off_h < H
    mask_w = off_w < W
    
    # Create output pointer offsets
    off_out = (off_b[:, None, None, None] * out_sB + 
               off_c[None, :, None, None] * out_sC + 
               off_h[None, None, :, None] * out_sH + 
               off_w[None, None, None, :] * out_sW)
    
    # Load in_3 (accumulator) - shape [B, C, H, J] but we need [B, C, H, W] after einsum
    # in_3 is [B, 512, 64, 64], same as final output
    off_in3_base = (off_b[:, None, None, None] * in_3_sB + 
                    off_c[None, :, None, None] * in_3_sC + 
                    off_h[None, None, :, None] * in_3_sH)
    # Initialize accumulator with in_3 values
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Load in_3
    for j_idx in range(BLOCK_SIZE_W):
        mask = mask_b[:, None, None, None] & mask_c[None, :, None, None] & mask_h[None, None, :, None]
        off_j = j_idx * in_3_sJ
        off_in3 = off_in3_base + off_j
        in3_val = tl.load(in_3_ptr + off_in3, mask=mask, other=0.0)
        acc = acc + in3_val
    
    # Compute einsum: sum over J of in_4[b,c,h,j] * in_1[b,h,w,j]
    # in_4: [B, C, H, J], in_1: [B, H, W, J], output: [B, C, H, W]
    
    # Load in_2 (bias) - shape [B, C, H, W]
    off_in2_base = (off_b[:, None, None, None] * in_2_sB + 
                    off_c[None, :, None, None] * in_2_sC + 
                    off_h[None, None, :, None] * in_2_sH)
    off_in2 = off_in2_base + off_w[None, None, None, :] * in_2_sW
    mask_in2 = mask_b[:, None, None, None] & mask_c[None, :, None, None] & mask_h[None, None, :, None] & mask_w[None, None, None, :]
    in2_val = tl.load(in_2_ptr + off_in2, mask=mask_in2, other=0.0)
    
    # Compute einsum contraction
    for j in range(J_BLOCK):
        # Load in_1 [B, H, W, J]
        off_in1 = (off_b[:, None, None] * in_1_sB + 
                   off_h[None, :, None] * in_1_sH + 
                   off_w[None, None, :] * in_1_sW + 
                   j * in_1_sJ)
        mask_in1 = mask_b[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
        in1_val = tl.load(in_1_ptr + off_in1, mask=mask_in1, other=0.0)
        
        # Load in_4 [B, C, H, J]
        off_in4 = (off_b[:, None, None] * in_4_sB + 
                   off_c[None, :, None] * in_4_sC + 
                   off_h[None, None, :] * in_4_sH + 
                   j * in_4_sJ)
        mask_in4 = mask_b[:, None, None] & mask_c[None, :, None] & mask_h[None, None, :]
        in4_val = tl.load(in_4_ptr + off_in4, mask=mask_in4, other=0.0)
        
        # Accumulate einsum result
        # in1_val: [B, H, W], in4_val: [B, C, H]
        # Need to broadcast properly
        # in4_val expanded: [B, C, 1, H] -> [B, C, H] via sum over J
        # Actually: in4_val[b,c,h] * in1_val[b,h,w]
        # Result: acc[b,c,h,w] += in4_val[b,c,h,j] * in1_val[b,h,w,j]
        acc = acc + in4_val[:, :, :, None] * in1_val[:, None, :, :]
    
    # Apply scale and add in_2
    result = acc * scale + in2_val
    
    # Store result
    tl.store(out_ptr + off_out, result, mask=mask_b[:, None, None, None] & mask_c[None, :, None, None] & mask_h[None, None, :, None] & mask_w[None, None, None, :])


@torch.fx.wrap
def fused_ccnet_wrapper(in_0, in_1, in_2, in_3, in_4):
    """
    Fused kernel for CCNet computation:
    - einsum: 'bchj,bhwj->bchw' (in_4, in_1)
    - in_3 += einsum
    - result = (in_3 * in_0) + in_2
    - contiguous
    """
    B, H, W, J = in_1.shape
    C = in_2.shape[1]
    scale = float(in_0.item() if isinstance(in_0, torch.Tensor) else in_0)
    
    # Allocate output
    out = torch.empty((B, C, H, W), dtype=in_2.dtype, device=in_2.device)
    
    # Block configuration
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_C = 8
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 16
    J_BLOCK = J  # 64
    
    # Grid dimensions
    grid_b = (B + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    
    fused_ccnet_kernel[(grid_b, grid_c, grid_h)](
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        in_4_ptr=in_4,
        out_ptr=out,
        B=B, C=C, H=H, W=W, J=J,
        scale=scale,
        in_1_sB=in_1.stride(0), in_1_sH=in_1.stride(1), in_1_sW=in_1.stride(2), in_1_sJ=in_1.stride(3),
        in_2_sB=in_2.stride(0), in_2_sC=in_2.stride(1), in_2_sH=in_2.stride(2), in_2_sW=in_2.stride(3),
        in_3_sB=in_3.stride(0), in_3_sC=in_3.stride(1), in_3_sH=in_3.stride(2), in_3_sJ=in_3.stride(3),
        in_4_sB=in_4.stride(0), in_4_sC=in_4.stride(1), in_4_sH=in_4.stride(2), in_4_sJ=in_4.stride(3),
        out_sB=out.stride(0), out_sC=out.stride(1), out_sH=out.stride(2), out_sW=out.stride(3),
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        J_BLOCK=J_BLOCK,
    )
    
    return out.contiguous()


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the CCNet computation pattern:
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    in_3 += einsum (in-place add)
    tmp_3 = in_5 * in_0 (where in_5 = in_3 after +=)
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return (tmp_5,)
    """
    einsum = torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)
    # Use new variable name to avoid conflict with input parameter
    acc = in_3 + einsum
    tmp_3 = acc * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_ccnet_wrapper