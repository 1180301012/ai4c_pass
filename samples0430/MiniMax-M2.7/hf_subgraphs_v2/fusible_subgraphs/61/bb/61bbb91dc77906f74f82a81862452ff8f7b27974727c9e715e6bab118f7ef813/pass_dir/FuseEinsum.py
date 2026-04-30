import torch
import triton
import triton.language as tl

@triton.jit
def optimized_einsum_kernel(
    in_1_ptr, in_4_ptr, out_ptr,
    B, H, W, J, C,
    in_1_sB, in_1_sH, in_1_sW, in_1_sJ,
    in_4_sB, in_4_sC, in_4_sH, in_4_sJ,
    out_sB, out_sC, out_sH, out_sW,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    J_BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    off_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    off_w = tl.arange(0, BLOCK_SIZE_W)
    
    mask_b = off_b < B
    mask_c = off_c < C
    mask_h = off_h < H
    mask_w = off_w < W
    
    off_out = (off_b[:, None, None, None] * out_sB + 
               off_c[None, :, None, None] * out_sC + 
               off_h[None, None, :, None] * out_sH + 
               off_w[None, None, None, :] * out_sW)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    for j in range(J_BLOCK):
        off_in1 = (off_b[:, None, None] * in_1_sB + 
                   off_h[None, :, None] * in_1_sH + 
                   off_w[None, None, :] * in_1_sW + 
                   j * in_1_sJ)
        mask_in1 = mask_b[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
        in1_val = tl.load(in_1_ptr + off_in1, mask=mask_in1, other=0.0)
        
        off_in4 = (off_b[:, None, None] * in_4_sB + 
                   off_c[None, :, None] * in_4_sC + 
                   off_h[None, None, :] * in_4_sH + 
                   j * in_4_sJ)
        mask_in4 = mask_b[:, None, None] & mask_c[None, :, None] & mask_h[None, None, :]
        in4_val = tl.load(in_4_ptr + off_in4, mask=mask_in4, other=0.0)
        
        acc = acc + in4_val[:, :, :, None] * in1_val[:, None, :, :]
    
    tl.store(out_ptr + off_out, acc, 
             mask=mask_b[:, None, None, None] & mask_c[None, :, None, None] & mask_h[None, None, :, None] & mask_w[None, None, None, :])


@torch.fx.wrap
def optimized_einsum_wrapper(in_1, in_4):
    """
    Optimized einsum for 'bchj,bhwj->bchw' pattern.
    in_1: [B, H, W, J]
    in_4: [B, C, H, J]
    out:  [B, C, H, W]
    """
    B, H, W, J = in_1.shape
    C = in_4.shape[1]
    
    out = torch.empty((B, C, H, W), dtype=in_4.dtype, device=in_4.device)
    
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_C = 8
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 16
    J_BLOCK = J
    
    grid_b = (B + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = (H + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    
    optimized_einsum_kernel[(grid_b, grid_c, grid_h)](
        in_1_ptr=in_1, in_4_ptr=in_4, out_ptr=out,
        B=B, H=H, W=W, J=J, C=C,
        in_1_sB=in_1.stride(0), in_1_sH=in_1.stride(1), in_1_sW=in_1.stride(2), in_1_sJ=in_1.stride(3),
        in_4_sB=in_4.stride(0), in_4_sC=in_4.stride(1), in_4_sH=in_4.stride(2), in_4_sJ=in_4.stride(3),
        out_sB=out.stride(0), out_sC=out.stride(1), out_sH=out.stride(2), out_sW=out.stride(3),
        BLOCK_SIZE_B=BLOCK_SIZE_B, BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        J_BLOCK=J_BLOCK,
    )
    
    return out


def pattern(in_1, in_4):
    """Match einsum 'bchj,bhwj->bchw' pattern."""
    return torch.functional.einsum('bchj,bhwj->bchw', in_4, in_1)


def replacement_args(in_1, in_4):
    return (in_1, in_4)


def replacement_func():
    return optimized_einsum_wrapper