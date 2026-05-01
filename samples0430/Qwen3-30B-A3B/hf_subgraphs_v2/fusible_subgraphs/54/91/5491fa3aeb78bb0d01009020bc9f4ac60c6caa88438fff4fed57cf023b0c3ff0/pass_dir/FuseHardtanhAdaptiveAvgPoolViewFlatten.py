import torch
import triton
import triton.language as tl

def pattern(x):
    t0 = torch.nn.functional.hardtanh(x, 0.0, 6.0, True)
    t1 = torch.nn.functional.adaptive_avg_pool2d(t0, (1,1))
    t2 = t1.view(4, -1)
    t3 = torch.flatten(t2, 1)
    return t3

def replacement_args(x):
    return (x,)

@triton.jit
def fused_avg_pool_relu6_kernel(
    in_ptr, out_ptr,
    B, C, H, W,
    BLOCK_B: tl.constexpr = 16,
    BLOCK_C: tl.constexpr = 16
):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    offset = (b_idx * C + c_idx) * H * W
    sum_val = tl.zeros((), dtype=tl.float32)
    for h in range(H):
        for w in range(W):
            idx = offset + h * W + w
            val = tl.load(in_ptr + idx)
            clipped = tl.clamp(val, 0.0, 6.0)
            sum_val = sum_val + clipped
    mean_val = sum_val / (H * W)
    out_idx = b_idx * C + c_idx
    tl.store(out_ptr + out_idx, mean_val)

@torch.fx.wrap
def fused_avg_pool_relu6(x):
    B, C, H, W = x.shape
    x_fp32 = x.to(torch.float32)
    out_fp32 = torch.empty((B, C), dtype=torch.float32, device=x.device)
    fused_avg_pool_relu6_kernel[(B, C)](
        x_fp32, out_fp32, B, C, H, W, BLOCK_B=16, BLOCK_C=16
    )
    return out_fp32.to(x.dtype)

def replacement_func():
    return fused_avg_pool_relu6