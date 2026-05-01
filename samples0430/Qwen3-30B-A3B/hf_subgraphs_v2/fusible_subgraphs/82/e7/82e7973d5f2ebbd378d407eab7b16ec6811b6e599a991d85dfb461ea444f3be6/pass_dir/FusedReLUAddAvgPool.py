import torch
import triton
import triton.language as tl

# Pattern matching function
@torch.fx.wrap
def pattern(in_0, in_1):
    relu_out = torch.nn.functional.relu(in_1, inplace=False)
    add_out = relu_out + in_0
    pool_out = torch.nn.functional.adaptive_avg_pool2d(add_out, 1)
    return pool_out

# Argument extraction function
@torch.fx.wrap
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Fused kernel implementation
@triton.jit
def fused_avg_pool_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr
):
    block_id = tl.program_id(0)
    b = block_id // C
    c = block_id % C

    in0_offset = b * C * H * W + c * H * W
    in1_offset = in0_offset
    out_offset = b * C + c

    # Thread 0 handles all spatial elements for this channel
    if tl.thread_id(0) == 0:
        sum_val = 0.0
        for h in range(H):
            for w in range(W):
                idx = h * W + w
                in0_val = tl.load(in0_ptr + in0_offset + idx)
                in1_val = tl.load(in1_ptr + in1_offset + idx)
                relu_val = in1_val if in1_val > 0.0 else 0.0
                sum_val += relu_val + in0_val
        avg = sum_val / (H * W)
        tl.store(out_ptr + out_offset, avg)

# Kernel wrapper
@torch.fx.wrap
def fused_avg_pool_wrapper(in_0, in_1):
    B, C, H, W = in_0.shape
    out = torch.empty((B, C), dtype=in_0.dtype, device=in_0.device)
    num_blocks = B * C
    fused_avg_pool_kernel[(num_blocks,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        out_ptr=out,
        B=B, C=C, H=H, W=W,
        BLOCK_SIZE=128
    )
    return out.reshape(B, C, 1, 1)

# Replacement function
def replacement_func():
    return fused_avg_pool_wrapper