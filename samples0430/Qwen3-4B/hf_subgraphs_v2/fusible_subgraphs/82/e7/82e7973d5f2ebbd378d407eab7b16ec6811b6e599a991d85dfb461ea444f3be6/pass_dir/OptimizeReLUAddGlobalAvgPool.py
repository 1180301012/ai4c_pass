import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Return output structure that matches the model's return
    return (torch.empty(1),)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def triton_global_avg_pool_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Process each batch-channel pair
    b = tl.program_id(0)
    c = tl.program_id(1)
    # Compute spatial average
    total = 0.0
    for h in range(H):
        for w in range(W):
            idx = b * C + c
            elem_idx = idx * (H * W) + h * W + w
            in_0_val = tl.load(in_0_ptr + elem_idx)
            in_1_val = tl.load(in_1_ptr + elem_idx)
            relu_val = tl.where(in_1_val > 0, in_1_val, 0.0)
            total += relu_val + in_0_val
    out_val = total / (H * W)
    tl.store(out_ptr + b * C + c, out_val)

@torch.fx.wrap
def triton_global_avg_pool(in_0, in_1):
    B = in_0.shape[0]
    C = in_0.shape[1]
    H = in_0.shape[2]
    W = in_0.shape[3]
    out = torch.zeros((B, C), dtype=in_0.dtype, device=in_0.device)
    triton_global_avg_pool_kernel[
        (B, C)
    ](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        B=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=1024,
    )
    return (out.unsqueeze(-1).unsqueeze(-1),)

def replacement_func():
    return triton_global_avg_pool