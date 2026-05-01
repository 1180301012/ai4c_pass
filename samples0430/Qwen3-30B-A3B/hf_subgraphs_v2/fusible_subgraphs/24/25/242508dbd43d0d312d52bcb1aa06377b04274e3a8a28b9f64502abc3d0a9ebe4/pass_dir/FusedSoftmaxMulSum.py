import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_softmax_sum_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    H,
    W,
    D,
    BLOCK_D: tl.constexpr
):
    h = tl.program_id(0)
    w = tl.program_id(1)
    d = tl.thread_id(0)
    
    if h >= H or w >= W or d >= D:
        return
    
    # Load in_1 values for current depth d (two channels)
    in_1_0 = tl.load(in_1_ptr + d)
    in_1_1 = tl.load(in_1_ptr + 256 + d)
    
    # Compute softmax for two-element vector
    max_val = tl.maximum(in_1_0, in_1_1)
    exp0 = tl.exp(in_1_0 - max_val)
    exp1 = tl.exp(in_1_1 - max_val)
    denom = exp0 + exp1
    softmax0 = exp0 / denom
    softmax1 = exp1 / denom
    
    # Calculate offsets for in_0
    offset = d * (H * W) + h * W + w
    in_0_0 = tl.load(in_0_ptr + offset)
    in_0_1 = tl.load(in_0_ptr + offset + 256 * H * W)
    
    # Compute sum over channel dimension (c=0 and c=1)
    res = in_0_0 * softmax0 + in_0_1 * softmax1
    
    # Store result to output (shape [batch, depth, H, W])
    out_offset = d * (H * W) + h * W + w
    tl.store(out_ptr + out_offset, res)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    batch, channels, depth, H, W = in_0.shape
    out = torch.empty(batch, depth, H, W, dtype=in_0.dtype, device=in_0.device)
    BLOCK_D = 256
    grid = (H, W)
    
    # Launch kernel
    fused_softmax_sum_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        H=H,
        W=W,
        D=depth,
        BLOCK_D=BLOCK_D
    )
    
    return out

def replacement_func():
    return kernel_wrapper