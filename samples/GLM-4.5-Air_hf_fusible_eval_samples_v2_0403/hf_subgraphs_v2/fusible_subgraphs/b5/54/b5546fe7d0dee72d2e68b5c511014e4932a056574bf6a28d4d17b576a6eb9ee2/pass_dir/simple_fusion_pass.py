import torch
import triton
import triton.language as tl

def pattern(in_6, tmp_5, in_4, scalar):
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = scalar * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    return tmp_10

def replacement_args(in_6, tmp_5, in_4, scalar):
    return (in_6, tmp_5, in_4, scalar)

@triton.jit
def simple_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)
    
    out = x + y
    tl.store(out_ptr + offset, out, mask=mask)

@torch.fx.wrap
def simple_fusion_ops(in_6, tmp_5, in_4, scalar):
    N, C, H, W = tmp_5.shape
    
    # Element-wise multiplication: in_6 * tmp_5
    multiplied = in_6 * tmp_5
    
    # Pad the result 
    padded = torch.nn.functional.pad(multiplied, (0, 0, 1, 0, 0, 0), 'constant', value=0.0)
    
    # Scalar multiplication: scalar * in_4
    scaled = scalar * in_4
    
    # Add padded and scaled
    total_elements = N * C * H * W
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Prepare output
    added = torch.empty_like(padded, dtype=padded.dtype)
    
    simple_add_kernel[grid_size](
        padded.contiguous().data_ptr(),
        scaled.contiguous().data_ptr(),
        added.data_ptr(),
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Transpose dimensions 1 and 2
    result = added.transpose(1, 2)
    
    return result

def replacement_func():
    return simple_fusion_ops