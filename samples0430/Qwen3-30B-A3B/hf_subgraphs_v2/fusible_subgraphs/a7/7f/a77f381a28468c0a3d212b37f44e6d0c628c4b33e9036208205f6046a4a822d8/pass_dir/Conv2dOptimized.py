import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (0, 0), (1, 1), 1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 2048, None), slice(None, None, None), slice(None, None, None))]
    return (tmp_2, conv2d)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def conv2d_kernel(
    X_ptr,  # Input tensor
    W_ptr,  # Weight tensor
    Y_ptr,  # Output tensor
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    in_height,  # Input height
    in_width,  # Input width
    out_height,  # Output height
    out_width,  # Output width
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    pid = tl.program_id(0)
    block_batch = pid // (out_channels * out_height * out_width)
    block_oc = (pid // (out_height * out_width)) % out_channels
    block_hw = pid % (out_height * out_width)
    block_h = block_hw // out_width
    block_w = block_hw % out_width
    
    in_h_start = block_h * 2
    in_w_start = block_w * 2
    
    acc = tl.zeros((BLOCK_IC,), dtype=tl.float32)
    
    for ic_start in range(0, in_channels, BLOCK_IC):
        input_data = tl.load(
            X_ptr + block_batch * in_channels * in_height * in_width + ic_start * in_height * in_width + in_h_start * in_width + in_w_start,
            shape=(BLOCK_IC, BLOCK_H, BLOCK_W),
            mask=(ic_start + tl.arange(0, BLOCK_IC) < in_channels)[:, None, None],
            other=0.0
        )
        
        weights = tl.load(
            W_ptr + block_oc * in_channels + ic_start,
            shape=(BLOCK_IC,),
            mask=ic_start + tl.arange(0, BLOCK_IC) < in_channels,
            other=0.0
        )
        
        acc += tl.dot(weights, input_data.reshape(BLOCK_IC))
    
    tl.store(
        Y_ptr + block_batch * out_channels * out_height * out_width + block_oc * out_height * out_width + block_h * out_width + block_w,
        acc[0].to(tl.float32)
    )

@torch.fx.wrap
def conv2d_optimized(in_0, in_1):
    batch_size = in_1.shape[0]
    in_channels = in_1.shape[1]
    in_height = in_1.shape[2]
    in_width = in_1.shape[3]
    out_channels = in_0.shape[0]
    
    out_height = (in_height + 2 * 0 - 1 * (2 - 1) - 1) // 2 + 1
    out_width = (in_width + 2 * 0 - 1 * (2 - 1) - 1) // 2 + 1
    
    out = torch.empty((batch_size, out_channels, out_height, out_width), dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_BATCH = 1
    BLOCK_OC = 64
    BLOCK_IC = 64
    BLOCK_H = 1
    BLOCK_W = 1
    
    num_programs = out_channels * out_height * out_width
    
    conv2d_kernel[(num_programs,)](
        X_ptr=in_1,
        W_ptr=in_0,
        Y_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return out

def replacement_func():
    return conv2d_optimized