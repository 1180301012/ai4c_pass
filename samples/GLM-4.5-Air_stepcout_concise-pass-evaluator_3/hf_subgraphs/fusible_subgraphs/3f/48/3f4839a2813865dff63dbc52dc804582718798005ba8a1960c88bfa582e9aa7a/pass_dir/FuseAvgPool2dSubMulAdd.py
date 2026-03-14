import torch
import triton
import triton.language as tl


# Pattern can use torch operations for matching
def pattern(in_0, in_1, in_2):
    pooled = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
    diff = pooled - in_2
    tmp_4 = in_0.unsqueeze(-1)
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_6 = tmp_5 * diff
    tmp_7 = in_2 + tmp_6
    tmp_8 = in_1.unsqueeze(-1)
    tmp_9 = tmp_8.unsqueeze(-1)
    return tmp_7, tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Triton kernel for the fused computation
@triton.jit
def fused_kernel(
    in_ptr, scale_ptr, out_ptr,
    stride_b, stride_c, stride_h, stride_w,
    scale_stride,
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one batch and channel
    pid = tl.program_id(0)
    if pid >= B * C:
        return
    
    batch_idx = pid // C
    channel_idx = pid % C
    
    # Calculate offsets
    in_offset = batch_idx * stride_b + channel_idx * stride_c
    scale_offset = channel_idx * scale_stride
    
    # Load scale for this channel
    scale = tl.load(scale_ptr + scale_offset).to(tl.float32)
    
    # Compute output for each spatial position
    out_offset = in_offset
    
    for h in range(H):
        for w in range(W):
            # Compute 3x3 average pool manually
            pooled = 0.0
            count = 0.0
            
            for kh in range(-1, 2):
                for kw in range(-1, 2):
                    h_in = h + kh
                    w_in = w + kw
                    if 0 <= h_in < H and 0 <= w_in < W:
                        idx = in_offset + h_in * stride_h + w_in * stride_w
                        pooled += tl.load(in_ptr + idx).to(tl.float32)
                        count += 1.0
            
            pooled = pooled / count
            
            # Load original
            orig_idx = in_offset + h * stride_h + w * stride_w
            orig = tl.load(in_ptr + orig_idx).to(tl.float32)
            
            # Compute: out = orig + scale * (pooled - orig)
            out_val = orig + scale * (pooled - orig)
            
            # Store
            out_idx = out_offset + h * stride_h + w * stride_w
            tl.store(out_ptr + out_idx, out_val)


@torch.fx.wrap
def replacement_func(in_0, in_1, in_2):
    """Fused kernel using Triton - no blocked torch APIs"""
    B, C, H, W = in_2.shape
    
    # Allocate output
    out = torch.empty_like(in_2)
    
    # Launch config
    num_programs = B * C
    BLOCK_SIZE = 64
    
    fused_kernel[(num_programs,)](
        in_ptr=in_2,
        scale_ptr=in_0,
        out_ptr=out,
        stride_b=in_2.stride(0),
        stride_c=in_2.stride(1),
        stride_h=in_2.stride(2),
        stride_w=in_2.stride(3),
        scale_stride=in_0.stride(0) if in_0.dim() > 0 else 1,
        B=B, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Second output - just unsqueeze
    tmp_8 = in_1.unsqueeze(-1)
    tmp_9 = tmp_8.unsqueeze(-1)
    
    return out, tmp_9


def replacement_func():
    return replacement_func_impl