import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    hardswish = torch.nn.functional.hardswish(conv2d, True)
    flat = hardswish.flatten(1, -1)
    return flat

def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0)

@triton.jit
def fused_conv_hardswish_kernel(
    in_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    batch,
    in_channels,
    out_channels,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N
    
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < out_channels
    mask_n = offs_n < batch
    
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    
    for k in range(0, in_channels, 64):
        k_end = min(k + 64, in_channels)
        mask_k = (k + tl.arange(0, 64) < k_end)
        
        input_data = tl.load(
            in_ptr + offs_n[:, None] * in_channels + k + tl.arange(0, 64),
            mask=(mask_n[:, None] & mask_k[None, :]),
            other=0.0,
        )
        
        mask_k = (k + tl.arange(0, 64) < k_end)
        weights_data = tl.load(
            w_ptr + (k + tl.arange(0, 64)[:, None]) * out_channels + offs_m[None, :],
            mask=(mask_k[:, None] & mask_m[None, :]),
            other=0.0,
        )
        
        acc += tl.dot(input_data, weights_data, allow_tf32=True)
    
    bias = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0)
    output = acc + bias[None, :]
    
    # Hardswish: output = output * (output + 3) / 6
    output = output * (output + 3.0) / 6.0
    
    tl.store(
        out_ptr + offs_n[:, None] * out_channels + offs_m[None, :],
        output,
        mask=(mask_n[:, None] & mask_m[None, :])
    )

@torch.fx.wrap
def fused_conv_hardswish_wrapper(in_2, in_1, in_0):
    batch, in_channels, H, W = in_2.shape
    out_channels = in_1.shape[0]
    
    assert H == 1 and W == 1, "Expected 1x1 spatial dimensions"
    
    in_flat = in_2
    w_flat = in_1
    
    out = torch.empty((batch, out_channels), dtype=in_2.dtype, device=in_2.device)
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    grid_m = (out_channels + BLOCK_M - 1) // BLOCK_M
    grid_n = (batch + BLOCK_N - 1) // BLOCK_N
    
    fused_conv_hardswish_kernel[(grid_m, grid_n)](
        in_flat,
        w_flat,
        in_0,
        out,
        batch,
        in_channels,
        out_channels,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return out

def replacement_func():
    return fused_conv_hardswish_wrapper