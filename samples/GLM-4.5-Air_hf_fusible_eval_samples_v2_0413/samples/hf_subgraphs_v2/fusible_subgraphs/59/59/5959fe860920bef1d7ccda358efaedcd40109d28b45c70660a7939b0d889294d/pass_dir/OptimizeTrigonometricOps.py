import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    return tmp_1.cos().to(torch.bfloat16), tmp_1.sin().to(torch.bfloat16)

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def fused_trigonometric_kernel(freqs_ptr, cos_out_ptr, sin_out_ptr, n_elements, original_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input frequencies (original, half-size tensor)
    freqs = tl.load(freqs_ptr + (offsets // 2), mask=(offsets // 2) < original_elements, other=0.0)
    
    # Apply trigonometric functions
    cos_vals = tl.cos(freqs)
    sin_vals = tl.sin(freqs)
    
    # Convert to bfloat16
    cos_bf16 = tl.cast(cos_vals, tl.bfloat16)
    sin_bf16 = tl.cast(sin_vals, tl.bfloat16)
    
    # Store results - each value appears twice in the output (concatenated)
    tl.store(cos_out_ptr + offsets, cos_bf16, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_bf16, mask=mask)

@torch.fx.wrap
def fused_trigonometric_ops(in_1):
    # Get the original tensor shape and compute output shape after concatenation
    original_shape = in_1.shape
    concatenated_shape = list(original_shape)
    concatenated_shape[-1] *= 2  # Double the last dimension
    n_elements = in_1.numel() * 2
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate outputs with concatenated shape
    cos_out = torch.empty(concatenated_shape, dtype=torch.bfloat16, device=in_1.device)
    sin_out = torch.empty(concatenated_shape, dtype=torch.bfloat16, device=in_1.device)
    
    # Launch kernel with original (pre-concatenation) tensor
    fused_trigonometric_kernel[(num_programs,)](
        freqs_ptr=in_1,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_elements=n_elements,
        original_elements=in_1.numel(),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return cos_out, sin_out

def replacement_func():
    return fused_trigonometric_ops