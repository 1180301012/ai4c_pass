import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_mul_sum_sigmoid_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, C, H, W: tl.constexpr,
    stride_0_b, stride_0_c, stride_0_h, stride_0_w,
    stride_1_b, stride_1_c, stride_1_h, stride_1_w,
    BLOCK_H: tl.constexpr,
):
    # Grid: (B, H // BLOCK_H)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_off = tl.arange(0, W)  # Full width per program
    
    h_mask = h_off < H
    
    # Accumulate over C dimension  
    acc = tl.zeros([BLOCK_H, W], dtype=tl.float32)
    
    for c in range(C):
        off_0 = pid_b * stride_0_b + c * stride_0_c + h_off[:, None] * stride_0_h + w_off[None, :] * stride_0_w
        off_1 = pid_b * stride_1_b + c * stride_1_c + h_off[:, None] * stride_1_h + w_off[None, :] * stride_1_w
        v0 = tl.load(in_0_ptr + off_0, mask=h_mask[:, None], other=0.0).to(tl.float32)
        v1 = tl.load(in_1_ptr + off_1, mask=h_mask[:, None], other=0.0).to(tl.float32)
        acc += v0 * v1
    
    result = tl.sigmoid(acc)
    
    # Store to output [B, 1, H, W]
    out_off = pid_b * (H * W) + h_off[:, None] * W + w_off[None, :]
    tl.store(out_ptr + out_off, result, mask=h_mask[:, None])


@torch.fx.wrap
def fused_mul_sum_unsqueeze_sigmoid(in_0, in_1):
    B, C, H, W = in_0.shape
    out = torch.empty((B, 1, H, W), dtype=in_0.dtype, device=in_0.device)

    # Use BLOCK_H = 4 for more programs, or = H for fewer but bigger programs
    BLOCK_H = 4
    
    grid = (B, H // BLOCK_H)

    fused_mul_sum_sigmoid_kernel[grid](
        in_0, in_1, out,
        B, C, H, W,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        BLOCK_H=BLOCK_H,
    )

    return out


def replacement_func():
    return fused_mul_sum_unsqueeze_sigmoid