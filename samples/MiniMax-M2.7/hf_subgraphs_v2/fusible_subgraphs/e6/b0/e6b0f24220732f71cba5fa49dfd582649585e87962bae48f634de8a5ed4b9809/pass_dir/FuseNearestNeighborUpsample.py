import torch
import triton
import triton.language as tl


@triton.jit
def triton_concat_dim1_kernel(
    in_2_ptr, in_3_ptr, out_ptr,
    B: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
):
    """
    Optimized concatenation along dim 1.
    Concat [B, C2, H, W] + [B, C3, H, W] -> [B, C2+C3, H, W]
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    mask = (pid_h < H) and (pid_w < W)
    
    for c in range(C2 + C3):
        out_offset = pid_b * (C2 + C3) * H * W + c * H * W + pid_h * W + pid_w
        
        if c < C2:
            offset = pid_b * C2 * H * W + c * H * W + pid_h * W + pid_w
            val = tl.load(in_2_ptr + offset, mask=mask, other=0.0)
        else:
            offset = pid_b * C3 * H * W + (c - C2) * H * W + pid_h * W + pid_w
            val = tl.load(in_3_ptr + offset, mask=mask, other=0.0)
        
        tl.store(out_ptr + out_offset, val, mask=mask)


@torch.fx.wrap
def triton_concat_dim1(in_2, in_3):
    """Concatenate tensors along dim 1 using Triton."""
    B = in_2.shape[0]
    C2 = in_2.shape[1]
    C3 = in_3.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    
    output = torch.empty((B, C2 + C3, H, W), dtype=in_2.dtype, device=in_2.device)
    
    grid = (B, H, W)
    triton_concat_dim1_kernel[grid](
        in_2, in_3, output,
        B, C2, C3, H, W
    )
    return output


def pattern(in_2, in_3):
    """Match torch.cat along dim 1."""
    return torch.cat((in_2, in_3), 1)


def replacement_args(in_2, in_3):
    return (in_2, in_3)


def replacement_func():
    return triton_concat_dim1