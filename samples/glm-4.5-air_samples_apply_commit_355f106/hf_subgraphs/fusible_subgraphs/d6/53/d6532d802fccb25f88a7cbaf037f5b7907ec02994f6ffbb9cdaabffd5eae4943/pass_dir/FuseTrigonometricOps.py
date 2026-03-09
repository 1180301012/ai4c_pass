import torch
import triton
import triton.language as tl

def pattern(input_freqs):
    tmp_1 = torch.cat((input_freqs, input_freqs), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return tmp_6, tmp_7

def replacement_args(input_freqs):
    return (input_freqs,)

@triton.jit
def fused_trigonometric_kernel(input_ptr, out_cos_ptr, out_sin_ptr,
                              original_shape, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input (we only load half the elements since we duplicated in the kernel)
    original_elements = original_shape[1] * original_shape[2]
    half_offsets = offsets % original_elements
    input_val = tl.load(input_ptr + half_offsets, mask=mask, other=0.0)
    
    # Compute cos and sin together
    cos_val = tl.cos(input_val)
    sin_val = tl.sin(input_val)
    
    # Convert to bfloat16 and store
    tl.store(out_cos_ptr + offsets, cos_val.to(tl.bfloat16), mask=mask)
    tl.store(out_sin_ptr + offsets, sin_val.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def fused_trigonometric_ops(input_freqs):
    # Get original shape and determine final shape after concatenation
    original_shape = input_freqs.shape
    original_elements = input_freqs.numel()
    final_elements = original_elements * 2
    final_shape = (*original_shape[:-1], original_shape[-1] * 2)
    
    BLOCK_SIZE = 1024
    num_programs = (final_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out_cos = torch.empty(final_shape, dtype=torch.bfloat16, device=input_freqs.device)
    out_sin = torch.empty(final_shape, dtype=torch.bfloat16, device=input_freqs.device)
    
    fused_trigonometric_kernel[(num_programs,)](
        input_ptr=input_freqs,
        out_cos_ptr=out_cos,
        out_sin_ptr=out_sin,
        original_shape=original_shape,
        total_elements=final_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_cos, out_sin

def replacement_func():
    return fused_trigonometric_ops