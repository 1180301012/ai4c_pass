import torch
import triton
import triton.language as tl

# Simpler pattern - just match transpose
def pattern(x):
    return x.transpose(-2, -1)

def replacement_args(x):
    return (x,)

@triton.jit  
def triton_transpose_kernel(
    in_ptr, out_ptr,
    batch, channels, h, w,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for transpose(-2, -1)"""
    pid = tl.program_id(0)
    n_elements = batch * channels * h * w
    
    stride = tl.num_programs(0) * BLOCK_SIZE
    
    for idx in range(pid * BLOCK_SIZE, n_elements, stride):
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        tmp = offsets // (channels * h * w)
        b = tmp
        remaining = offsets % (channels * h * w)
        c = remaining // (h * w)
        rem2 = remaining % (h * w)
        h_idx = rem2 // w
        w_idx = rem2 % w
        
        src_idx = b * channels * h * w + c * h * w + h_idx * w + w_idx
        dst_idx = b * channels * w * h + c * w * h + w_idx * h + h_idx
        
        val = tl.load(in_ptr + src_idx, mask=mask, other=0.0)
        tl.store(out_ptr + dst_idx, val, mask=mask)


@torch.fx.wrap
def triton_transpose(x):
    """Wrapper for transpose kernel"""
    batch, channels, h, w = x.shape
    out = torch.empty((batch, channels, w, h), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    n_elements = batch * channels * h * w
    num_programs = min(65536, (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    triton_transpose_kernel[(num_programs,)](
        in_ptr=x,
        out_ptr=out,
        batch=batch,
        channels=channels,
        h=h,
        w=w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def replacement_func():
    return triton_transpose