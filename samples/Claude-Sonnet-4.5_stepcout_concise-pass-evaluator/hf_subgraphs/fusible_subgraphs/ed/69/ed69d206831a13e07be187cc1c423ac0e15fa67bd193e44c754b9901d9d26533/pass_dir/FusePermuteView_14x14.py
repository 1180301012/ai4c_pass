import torch
import triton
import triton.language as tl


def pattern(inp):
    """Pattern: permute + view for reshaping patches"""
    tmp_1 = inp.permute(0, 2, 1)
    tmp_2 = tmp_1.view(1, 384, 14, 14)
    return tmp_2


def replacement_args(inp):
    """Extract arguments for replacement"""
    return (inp,)


@triton.jit
def permute_reshape_kernel(
    inp_ptr,
    out_ptr,
    batch,
    seq_len,  # 196
    hidden,  # 384
    h,  # 14
    w,  # 14
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused permute(0, 2, 1) + view reshape
    Input: [batch, seq_len, hidden]
    After permute: [batch, hidden, seq_len]
    After view: [batch, hidden, h, w] where h*w == seq_len
    """
    pid = tl.program_id(0)
    
    total_elements = batch * seq_len * hidden
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load from input
    inp_val = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    
    # Compute input indices: [batch, seq_len, hidden]
    batch_idx = offsets // (seq_len * hidden)
    remainder = offsets % (seq_len * hidden)
    seq_idx = remainder // hidden
    hidden_idx = remainder % hidden
    
    # After permute(0, 2, 1): [batch, hidden, seq_len]
    # Output offset: batch * (hidden * seq_len) + hidden_idx * seq_len + seq_idx
    out_offset = batch_idx * (hidden * seq_len) + hidden_idx * seq_len + seq_idx
    
    # View doesn't change memory layout, just shape interpretation
    # So output goes to the same linear offset
    tl.store(out_ptr + out_offset, inp_val, mask=mask)


@torch.fx.wrap
def permute_reshape_14x14(inp):
    """Fused permute + reshape for 196 patches -> 14x14"""
    batch, seq_len, hidden = inp.shape
    
    # Hardcoded for this variant
    h = w = 14
    
    # Allocate output
    out = torch.empty((batch, hidden, h, w), device=inp.device, dtype=inp.dtype)
    
    # Launch kernel
    total_elements = batch * seq_len * hidden
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    permute_reshape_kernel[grid](
        inp,
        out,
        batch,
        seq_len,
        hidden,
        h,
        w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    """Return the replacement function"""
    return permute_reshape_14x14