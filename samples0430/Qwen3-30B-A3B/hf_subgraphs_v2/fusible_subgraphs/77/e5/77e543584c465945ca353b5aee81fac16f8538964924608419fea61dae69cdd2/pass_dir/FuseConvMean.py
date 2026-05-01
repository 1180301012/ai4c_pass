import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (0, 0), (0, 0), (0, 0), 1)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)

def replacement_args(in_0, in_1, stride, padding, dilation, groups, conv2d):
    return (conv2d,)

@triton.jit
def mean_reduction_kernel(
    x_ptr,
    y_ptr,
    B, C, H, W,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    batch = block_id // C
    channel = block_id % C
    
    accumulator = tl.zeros((1,), dtype=tl.float32)
    
    num_spatial = H * W
    start_idx = tl.thread_id(0) * (num_spatial // BLOCK_SIZE)
    end_idx = min(start_idx + (num_spatial // BLOCK_SIZE), num_spatial)
    
    for i in range(start_idx, end_idx):
        h = i // W
        w = i % W
        idx = batch * x_stride_b + channel * x_stride_c + h * x_stride_h + w * x_stride_w
        val = tl.load(x_ptr + idx)
        accumulator += val
    
    accumulator = tl.sum(accumulator)
    
    if tl.thread_id(0) == 0:
        total = H * W
        mean = accumulator / tl.cast(total, tl.float32)
        out_idx = batch * y_stride_b + channel * y_stride_c
        tl.store(y_ptr + out_idx, mean)

@torch.fx.wrap
def mean_reduction(x):
    B, C, H, W = x.shape
    x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
    y = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)
    y_stride_b, y_stride_c, _, _ = y.stride()
    
    grid = (B * C, 1)
    BLOCK_SIZE = 256
    
    mean_reduction_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        B=B, C=C, H=H, W=W,
        x_stride_b=x_stride_b, x_stride_c=x_stride_c, x_stride_h=x_stride_h, x_stride_w=x_stride_w,
        y_stride_b=y_stride_b, y_stride_c=y_stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y

def replacement_func():
    return mean_reduction