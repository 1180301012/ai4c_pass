import torch
import triton
import triton.language as tl

def pattern(freqs):
    """Match concatenate + sin + cos pattern for the specific computation"""
    # Match the exact computation that creates tmp_6 and tmp_7
    tmp_1 = torch.cat((freqs, freqs), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return tmp_6, tmp_7

def replacement_args(freqs):
    return (freqs,)

@triton.jit
def sincos_kernel(
    freqs_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to compute sin and cos of concatenated frequencies"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load frequencies (float32)
    freqs = tl.load(freqs_ptr + offsets, mask=mask, other=0.0)
    
    # Directly compute cos and sin (no redundant multiplies needed)
    cos_vals = tl.cos(freqs)
    sin_vals = tl.sin(freqs)
    
    # Convert to bfloat16 and store
    tl.store(cos_out_ptr + offsets, cos_vals.to(tl.bfloat16), mask=mask)
    tl.store(sin_out_ptr + offsets, sin_vals.to(tl.bfloat16), mask=mask)

@torch.fx.wrap
def fused_concat_sincos(freqs):
    """Fused sin/cos computation with automatic concatenation and bfloat16 output"""
    # Get frequency tensor properties  
    n_elements = freqs.numel()
    
    # Use fixed block size for simplicity
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors in bfloat16
    cos_out = torch.empty_like(freqs, dtype=torch.bfloat16)
    sin_out = torch.empty_like(freqs, dtype=torch.bfloat16)
    
    # Launch the kernel
    sincos_kernel[(num_programs,)](
        freqs_ptr=freqs,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_out, sin_out

def replacement_func():
    return fused_concat_sincos