import torch
import triton
import triton.language as tl

@triton.jit
def expand_reshape_kernel(
    in_ptr, out_ptr,
    stride_in_b, stride_in_h, stride_in_s, stride_in_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    out_b = offsets // (stride_out_h * stride_out_s * stride_out_d)
    remaining = offsets % (stride_out_h * stride_out_s * stride_out_d)
    out_h = remaining // (stride_out_s * stride_out_d)
    remaining = remaining % (stride_out_s * stride_out_d)
    out_s = remaining // stride_out_d
    out_d = remaining % stride_out_d
    
    out_h_val = out_h
    in_h = 0
    in_s = out_s
    in_d = out_d
    
    in_offsets = out_b * stride_in_b + in_h * stride_in_h + in_s * stride_in_s + in_d * stride_in_d
    val = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    
    out_offsets = out_b * stride_out_b + out_h * stride_out_h + out_s * stride_out_s + out_d * stride_out_d
    tl.store(out_ptr + out_offsets, val, mask=mask)

@torch.fx.wrap
def expand_reshape_impl(inp):
    n_elements = 1 * 8 * 3 * 256
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((1, 8, 3, 256), dtype=inp.dtype, device=inp.device)
    
    expand_reshape_kernel[(num_programs,)](
        in_ptr=inp,
        out_ptr=out,
        stride_in_b=inp.stride(0),
        stride_in_h=inp.stride(1),
        stride_in_s=inp.stride(2),
        stride_in_d=inp.stride(3),
        stride_out_b=out.stride(0),
        stride_out_h=out.stride(1),
        stride_out_s=out.stride(2),
        stride_out_d=out.stride(3),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(tmp_6):
    tmp_7 = tmp_6[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
    tmp_7 = None
    tmp_9 = tmp_8.reshape(1, 8, 3, 256)
    tmp_8 = None
    return tmp_9

def replacement_args(tmp_6):
    return (tmp_6,)

def replacement_func():
    return expand_reshape_impl