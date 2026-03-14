import torch
import triton
import triton.language as tl
import operator
import math

# This pass handles the 196 graph configuration:
# Graph: [1, 197, 384] -> split [1, 196] -> view [1, 384, 14, 14]

def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = torch.functional.split(tmp_0, [1, 196], 1)
    tmp_2 = operator.getitem(tmp_1, 0)
    tmp_3 = operator.getitem(tmp_1, 1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    tmp_5 = tmp_4.view(1, 384, 14, 14)
    return tmp_2, tmp_5

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Simple optimized kernel - single pass through memory
@triton.jit
def fused_add_split_transpose_simple_196(
    in_0_ptr, in_1_ptr, 
    out_0_ptr, out_1_ptr,
    hidden_dim, spatial_size, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple fused kernel:
    - First hidden_dim elements: add first row -> out_0
    - Remaining: add + transpose -> out_1
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    total_out1 = hidden_dim * spatial_size
    
    # First handle out_1 (the larger output with transpose)
    mask_out1 = offs < total_out1
    
    # Output index: offs = c * spatial_size + s
    c = offs // spatial_size
    s = offs % spatial_size
    
    # Input index: (s + 1) * hidden_dim + c (skip first row, transpose)
    in_idx = (s + 1) * hidden_dim + c
    
    val0 = tl.load(in_0_ptr + in_idx, mask=mask_out1, other=0.0)
    val1 = tl.load(in_1_ptr + in_idx, mask=mask_out1, other=0.0)
    tl.store(out_1_ptr + offs, val0 + val1, mask=mask_out1)
    
    # Handle out_0 (first row) - only first few thread blocks need this
    mask_out0 = offs < hidden_dim
    val0_first = tl.load(in_0_ptr + offs, mask=mask_out0, other=0.0)
    val1_first = tl.load(in_1_ptr + offs, mask=mask_out0, other=0.0)
    tl.store(out_0_ptr + offs, val0_first + val1_first, mask=mask_out0)

@torch.fx.wrap
def _fused_kernel_impl_196(in_0, in_1):
    """Kernel implementation"""
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]
    hidden_dim = in_0.shape[2]  # 384
    spatial_size = seq_len - 1  # 196
    
    # Output 0: [1, 1, 384]
    out_0 = torch.empty(batch_size, 1, hidden_dim, device=in_0.device, dtype=in_0.dtype)
    
    # Output 1: [1, 384, 14, 14] for 196 spatial
    hw = int(math.sqrt(spatial_size))  # 14
    out_1 = torch.empty(batch_size, hidden_dim, hw, hw, device=in_0.device, dtype=in_0.dtype)
    
    # Single kernel launch
    BLOCK_SIZE = 1024
    total_elements = hidden_dim * spatial_size
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_add_split_transpose_simple_196[grid](
        in_0, in_1, out_0, out_1,
        hidden_dim, spatial_size, seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_0, out_1)

def fused_replacement_196(in_0, in_1):
    """Wrapper that unpacks the tuple to match pattern return structure"""
    result = _fused_kernel_impl_196(in_0, in_1)
    return operator.getitem(result, 0), operator.getitem(result, 1)

def replacement_func():
    return fused_replacement_196