import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    tmp_3 = in_0[:, 1]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    tmp_7 = in_0[:, 2]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    return torch.cat((tmp_2, tmp_6, tmp_10), 1)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in_0_ptr: tl.pointer,
    in_1_ptr: tl.pointer,
    out_ptr: tl.pointer,
    B: tl.int32,
    H: tl.int32,
    W: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate our current thread's position
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    
    # Get values from inputs
    in_1_val = tl.load(in_1_ptr + b*H*W + h*W + w, mask=tl.ones(1), other=0.0)
    in_0_ch1 = tl.load(in_0_ptr + b*H*W + h*W + w, mask=tl.ones(1), other=0.0)
    in_0_ch2 = tl.load(in_0_ptr + b*H*W + h*W + w, mask=tl.ones(1), other=0.0)
    
    # Process and store results
    out_val1 = in_1_val * 0.458 + -0.030000000000000027
    out_val2 = in_0_ch1 * 0.448 + -0.08799999999999997
    out_val3 = in_0_ch2 * 0.45 + -0.18799999999999994

    tl.store(out_ptr + (b*3*H*W + h*W + w), out_val1)
    tl.store(out_ptr + (b*3*H*W + h*W + w + H*W), out_val2)
    tl.store(out_ptr + (b*3*H*W + h*W + w + 2*H*W), out_val3)

def triton_chan_kernel(in_0, in_1):
    B = in_0.shape[0]
    H = in_0.shape[2]
    W = in_0.shape[3]
    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)
    
    grid = (tl.cdiv(B*H*W, 1024),)
    optimized_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        B=B,
        H=H,
        W=W,
        BLOCK_SIZE=1024,
    )
    return out

def replacement_func():
    return triton_chan_kernel