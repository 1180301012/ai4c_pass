import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern optimized with Triton kernel for GPU performance
    Matches slice + expand + None dimensions operations
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = tmp_1[slice(None, None, None), slice(None, 7, None)]
    tmp_1 = None
    tmp_3 = tmp_2.expand(2, 7)
    tmp_2 = None
    tmp_4 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_0 = None
    return (tmp_3, tmp_4)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def triton_fusion_kernel(
    in1_ptr, out3_ptr,
    in1_batch, in1_seq,
    out3_batch, out3_seq,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for slice-expand fusion"""
    pid = tl.program_id(0)
    
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (out3_batch * out3_seq)
    
    if tl.any(mask):
        row = offsets // out3_seq
        col = offsets % out3_seq
        
        row_mask = (row < out3_batch) & mask
        if tl.any(row_mask):
            col_valid = col[row_mask]
            input_idx = col_valid
            output_idx = offsets[row_mask]
            
            input_val = tl.load(in1_ptr + input_idx, other=0)
            tl.store(out3_ptr + output_idx, input_val)

@torch.fx.wrap
def triton_optimized_forward(in_0, in_1):
    """High-performance Triton-optimized implementation"""
    in0_shape = in_0.shape
    in1_shape = in_1.shape
    
    out3_shape = (2, 7)
    out4_shape = (2, 1, 1, 7)
    
    out3 = torch.empty(out3_shape, dtype=in_1.dtype, device=in_1.device)
    out4 = torch.empty(out4_shape, dtype=in_0.dtype, device=in_0.device)
    
    total_elements = out3_shape[0] * out3_shape[1]
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    triton_fusion_kernel[grid](
        in1_ptr=in_1,
        out3_ptr=out3,
        in1_batch=in1_shape[0],
        in1_seq=in1_shape[1],
        out3_batch=out3_shape[0],
        out3_seq=out3_shape[1],
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    result_4 = in_0[:, None, None, :]
    return (out3, result_4)

def replacement_func():
    return triton_optimized_forward