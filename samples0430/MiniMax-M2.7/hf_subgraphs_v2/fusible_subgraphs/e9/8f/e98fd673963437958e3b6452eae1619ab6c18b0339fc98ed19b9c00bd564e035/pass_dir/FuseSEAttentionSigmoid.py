import torch
import triton
import triton.language as tl

# Pattern matching: Matches sigmoid -> view -> expand -> mul
def pattern(in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    return tmp_3

def replacement_args(in_1, in_2):
    return (in_1, in_2)

@triton.jit
def sigmoid_impl(x):
    # Sigmoid using fp32 for numerical stability
    x_fp32 = x.to(tl.float32)
    return (1.0 / (1.0 + tl.exp(-x_fp32))).to(x.dtype)

@triton.jit
def fused_se_mul_kernel(
    in_1_ptr, in_2_ptr, out_ptr,
    C: tl.constexpr, HW: tl.constexpr
):
    # pid identifies a channel to process
    pid = tl.program_id(0)
    ch = pid
    
    if ch < C:
        # Load sigmoid weight for this channel
        sigmoid_weight = tl.load(in_2_ptr + ch)
        sigmoid_weight = sigmoid_impl(sigmoid_weight)
        
        # Process all HW elements for this channel
        for pos in range(HW):
            offset = ch * HW + pos
            val = tl.load(in_1_ptr + offset)
            out_val = val * sigmoid_weight
            tl.store(out_ptr + offset, out_val)

@torch.fx.wrap
def fused_sigmoid_expand_mul(in_1, in_2):
    B, C, H, W = in_1.shape
    HW = H * W
    
    out = torch.empty_like(in_1)
    
    # Launch one program per channel
    grid = (C,)
    
    fused_se_mul_kernel[grid](
        in_1, in_2, out,
        C, HW
    )
    
    return out

def replacement_func():
    return fused_sigmoid_expand_mul