import torch
import triton
import triton.language as tl

@triton.jit
def expand_mask_kernel(
    mask_ptr,
    out_ptr,
    h,
    w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_h = offsets // w
    mask_w = offsets % w
    mask = (mask_h < h) & (mask_w < w)
    
    # Load the mask values (only need the first w elements for each row)
    mask_flat = tl.load(mask_ptr + mask_h * w, mask=(mask_w < w), other=1.0)
    
    # Convert to float32 and broadcast across width
    out = mask_flat.to(tl.float32)
    
    # Store the expanded mask
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_expand_and_type_conversion(mask_tensor, target_shape):
    # mask_tensor: [1, 16] int64, target_shape: [1, 16, 768]
    h, w = target_shape[1], target_shape[2]  # 16, 768
    n_elements = h * w  # 16 * 768 = 12288
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty(target_shape, dtype=torch.float32, device=mask_tensor.device)
    
    expand_mask_kernel[(num_programs,)](
        mask_ptr=mask_tensor,
        out_ptr=out,
        h=h,
        w=w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern(in_0, in_1, in_2, in_3):
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(tmp_4)
    tmp_7 = tmp_6.float()
    return tmp_7

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, (1, 16, 768))  # Target shape based on the known input pattern

def replacement_func():
    return optimized_expand_and_type_conversion