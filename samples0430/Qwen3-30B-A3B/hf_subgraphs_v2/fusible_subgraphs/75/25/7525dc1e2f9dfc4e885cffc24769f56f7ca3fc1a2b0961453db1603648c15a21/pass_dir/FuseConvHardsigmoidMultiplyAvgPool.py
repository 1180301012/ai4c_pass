import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardsigmoid(conv2d, False)
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_conv_hardsigmoid_avgpool(
    in_3_ptr,
    in_1_ptr,
    in_2_ptr,
    in_0_ptr,
    out_ptr,
    batch,
    channels,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    if batch_id >= batch:
        return

    # Compute convolution for this batch
    conv_vals = tl.zeros((channels,), dtype=tl.float32)
    for c in range(channels):
        val = 0.0
        for in_ch in range(1024):
            in_3_val = tl.load(in_3_ptr + batch_id * 1024 + in_ch)
            in_1_val = tl.load(in_1_ptr + c * 1024 + in_ch)
            val += in_3_val * in_1_val
        val += tl.load(in_0_ptr + c)
        conv_vals[c] = val

    # Apply hardsigmoid
    hardsigmoid_vals = tl.clamp(0.5 * conv_vals + 0.5, 0.0, 1.0)

    # Sum over spatial dimensions for in_2
    sum_vals = tl.zeros((channels,), dtype=tl.float32)
    for c in range(channels):
        s = 0.0
        for h in range(H):
            for w in range(W):
                in_2_val = tl.load(in_2_ptr + batch_id * channels * H * W + c * H * W + h * W + w)
                s += in_2_val
        sum_vals[c] = s

    # Final computation
    for c in range(channels):
        out_val = hardsigmoid_vals[c] * sum_vals[c] / (H * W)
        tl.store(out_ptr + batch_id * channels + c, out_val)

@torch.fx.wrap
def fused_kernel(in_0, in_1, in_2, in_3):
    batch = in_2.shape[0]
    channels = in_1.shape[0]
    H = in_2.shape[2]
    W = in_2.shape[3]
    out = torch.empty(batch, channels, device=in_2.device, dtype=in_2.dtype)
    fused_conv_hardsigmoid_avgpool[(batch,)](
        in_3_ptr=in_3,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_0_ptr=in_0,
        out_ptr=out,
        batch=batch,
        channels=channels,
        H=H,
        W=W,
        BLOCK_SIZE=128
    )
    return out.view(batch, channels, 1, 1)

def replacement_func():
    return fused_kernel