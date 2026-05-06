import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    conv_out = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    hard_sigmoid = torch.nn.functional.hardsigmoid(conv_out, False)
    mul_out = in_2 * hard_sigmoid
    pool_out = torch.nn.functional.adaptive_avg_pool2d(mul_out, 1)
    flatten_out = pool_out.flatten(1, -1)
    dropout_out = torch.nn.functional.dropout(flatten_out, 0.0, False, False)
    return (dropout_out,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def optimized_kernel(\n    in_0_ptr,\n    in_1_ptr,\n    in_2_ptr,\n    in_3_ptr,\n    out_ptr,\n    batch_size,\n    channels,\n    H,\n    W,\n    BLOCK_SIZE: tl.constexpr,\n):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for i in range(BLOCK_SIZE):
        batch_idx = offset[i] // channels
        channel_idx = offset[i] % channels
        total = 0.0
        for h in range(H):
            for w in range(W):
                in_3_val = tl.load(in_3_ptr + (batch_idx * channels * H * W + channel_idx * H * W + h * W + w), dtype=tl.float32)
                hard_val = tl.where(in_3_val < -2, 0.0, tl.where(in_3_val > 2, 1.0, 0.2 * in_3_val + 0.5))
                in_2_val = tl.load(in_2_ptr + (batch_idx * channels * H * W + channel_idx * H * W + h * W + w), dtype=tl.float32)
                total += hard_val * in_2_val
        total /= (H * W)
        tl.store(out_ptr + offset[i], total)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    batch_size = in_2.shape[0]
    channels = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    out = torch.empty(batch_size * channels, device=in_2.device, dtype=in_2.dtype)
    BLOCK_SIZE = 256
    num_blocks = (batch_size * channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    optimized_kernel[(num_blocks,)](\n        in_0_ptr=in_0,\n        in_1_ptr=in_1,\n        in_2_ptr=in_2,\n        in_3_ptr=in_3,\n        out_ptr=out,\n        batch_size=batch_size,\n        channels=channels,\n        H=H,\n        W=W,\n        BLOCK_SIZE=BLOCK_SIZE,\n    )
    return out

def replacement_func():
    return kernel_wrapper