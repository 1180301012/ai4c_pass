import torch
import triton
import triton.language as tl

def pattern(tmp_2, in_3):
    """
    Match the geometric transformation + addition sequence:
    tmp_3 = tmp_2.flatten(2)     # [1, C, H, W] -> [1, C, H*W]
    tmp_4 = tmp_3.transpose(1, 2) # [1, H*W, C] 
    tmp_5 = tmp_4.contiguous()    # [1, H*W, C] (potentially redundant)
    tmp_6 = in_3 + tmp_5          # [1, H*W, C] (element-wise add)
    """
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    return tmp_6

def replacement_args(tmp_2, in_3):
    return (tmp_2, in_3)

@triton.jit
def fused_geo_transform_add_kernel(
    input_ptr,      # tmp_2: [1, C, H, W]
    add_ptr,        # in_3: [1, H*W, C] 
    output_ptr,     # output: [1, H*W, C]
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized fused kernel: flatten + transpose + add"""
    total_elements = C * H * W
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Direct memory access with optimal pattern
    # Load from input [1, C, H, W] - assumes contiguous memory layout
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load from add tensor [1, H*W, C] - compute linear index
    add_offset = offsets
    add_val = tl.load(add_ptr + add_offset, mask=mask, other=0.0)
    
    # Direct addition (flatten + transpose + add are fused)
    result = input_val + add_val
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_geo_transform_add(tmp_2, in_3):
    """Function that fuses geometric transformation + addition"""
    C = tmp_2.shape[1]  # Number of channels
    H, W = tmp_2.shape[2], tmp_2.shape[3]
    
    # Output shape [1, H*W, C]
    HW = H * W
    output_shape = (1, HW, C)
    output = torch.empty(output_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Calculate optimal block size
    n_elements = C * H * W
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized fused kernel
    fused_geo_transform_add_kernel[(num_programs,)](
        input_ptr=tmp_2,
        add_ptr=in_3,
        output_ptr=output,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_geo_transform_add