import torch
import triton
import triton.language as tl

@triton.jit
def expand_reshape_kernel(
    in_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # For [1, 8, 3, 256] output with [1, 1, 3, 256] input
    # Each output element (b, h, s, d) maps to input element (0, 0, s, d)
    out_b = offsets // (8 * 3 * 256)
    remaining = offsets % (8 * 3 * 256)
    out_h = remaining // (3 * 256)
    remaining = remaining % (3 * 256)
    out_s = remaining // 256
    out_d = remaining % 256
    
    # Input is always (0, 0, s, d) for output (b, h, s, d)
    in_offset = out_s * 256 + out_d
    val = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    
    out_offset = out_b * (8 * 3 * 256) + out_h * (3 * 256) + out_s * 256 + out_d
    tl.store(out_ptr + out_offset, val, mask=mask)

@torch.fx.wrap
def expand_reshape_fusion(inp):
    n_elements = 1 * 8 * 3 * 256
    BLOCK_SIZE = 512
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty((1, 8, 3, 256), dtype=inp.dtype, device=inp.device)
    
    expand_reshape_kernel[(num_programs,)](
        in_ptr=inp,
        out_ptr=out,
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
    return expand_reshape_fusion