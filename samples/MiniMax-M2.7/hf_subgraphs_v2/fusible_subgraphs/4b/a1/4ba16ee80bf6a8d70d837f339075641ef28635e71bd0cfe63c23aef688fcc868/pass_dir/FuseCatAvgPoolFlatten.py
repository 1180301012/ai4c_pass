import torch
import triton
import triton.language as tl


@triton.jit
def kernel_impl(in_0, in_1, in_2, in_3, out_ptr, 
                n0, n1, n2, n3, spatial, total_ch):
    """
    Simple kernel processing one channel per program.
    """
    pid = tl.program_id(0)
    
    if pid >= total_ch:
        return
    
    # Determine source tensor and local channel
    if pid < n0:
        ptr = in_0
        local_ch = pid
    elif pid < n0 + n1:
        ptr = in_1
        local_ch = pid - n0
    elif pid < n0 + n1 + n2:
        ptr = in_2
        local_ch = pid - n0 - n1
    else:
        ptr = in_3
        local_ch = pid - n0 - n1 - n2
    
    # Compute base offset
    base = local_ch * spatial
    
    # Load and sum spatial values
    offsets = tl.arange(0, 32)
    mask = offsets < spatial
    
    vals = tl.load(ptr + base + offsets, mask=mask, other=0.0)
    total = tl.sum(vals)
    
    # Compute average
    result = total / tl.cast(spatial, tl.float32)
    
    # Convert dtype
    if out_ptr.dtype == tl.float16:
        result = result.to(tl.float16)
    elif out_ptr.dtype == tl.bfloat16:
        result = result.to(tl.bfloat16)
    
    # Store
    tl.store(out_ptr + pid, result)


@torch.fx.wrap
def fused_cat_avgpool_flatten(in_0, in_1, in_2, in_3):
    """
    Fused implementation of:
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    """
    c0, c1, c2, c3 = in_0.shape[1], in_1.shape[1], in_2.shape[1], in_3.shape[1]
    h, w = in_0.shape[2], in_0.shape[3]
    spatial_size = h * w
    total_channels = c0 + c1 + c2 + c3
    
    output = torch.empty((1, total_channels), dtype=in_0.dtype, device=in_0.device)
    
    kernel_impl[(total_channels,)](
        in_0, in_1, in_2, in_3,
        output,
        c0, c1, c2, c3,
        spatial_size,
        total_channels
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3):
    """Match the pattern: cat + adaptive_avg_pool2d + dropout + flatten"""
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_cat_avgpool_flatten