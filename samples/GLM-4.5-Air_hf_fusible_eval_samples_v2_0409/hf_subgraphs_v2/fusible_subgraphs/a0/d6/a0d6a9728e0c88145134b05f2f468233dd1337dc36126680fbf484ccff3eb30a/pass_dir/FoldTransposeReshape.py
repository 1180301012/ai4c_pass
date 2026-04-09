import torch
import triton
import triton.language as tl

def pattern(bmm_1):
    tmp_4 = bmm_1.view(1, bmm_1.shape[1], 1, bmm_1.shape[2])  # Generic view
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, tmp_5.shape[2] * tmp_5.shape[3])
    return tmp_4, tmp_5, tmp_6

def replacement_args(bmm_1):
    return (bmm_1,)

@triton.jit
def direct_reshape_kernel(x_ptr, out_ptr, H, D, BLOCK_SIZE: tl.constexpr):
    # Directly reshape from [1, H, 1, D] to [1, 1, H*D]
    # Process each element in the flattened [H, D] space
    elem_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = elem_idx < (H * D)
    
    # Load from [1, H, 1, D] layout: offset is H*D (batch) + row*D + col
    # Since we have shape [1, H, 1, D], we can index directly as [H*D]
    x = tl.load(x_ptr + elem_idx, mask=mask, other=0.0)
    
    # Store in [1, 1, H*D] layout
    tl.store(out_ptr + elem_idx, x, mask=mask)

@torch.fx.wrap
def direct_reshape_func(bmm_1):
    # Create the same view operation as in the pattern
    tmp_4 = bmm_1.view(1, bmm_1.shape[1], 1, bmm_1.shape[2])
    H = tmp_4.shape[1]
    D = tmp_4.shape[3]
    total_elements = H * D
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output should be [1, 1, H*D]
    out_reshaped = torch.empty(1, 1, total_elements, dtype=tmp_4.dtype, device=tmp_4.device)
    
    direct_reshape_kernel[(num_programs,)](
        x_ptr=tmp_4.data_ptr(),
        out_ptr=out_reshaped.data_ptr(),
        H=H, D=D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_4, tmp_4.transpose(1, 2), out_reshaped

def replacement_func():
    return direct_reshape_func