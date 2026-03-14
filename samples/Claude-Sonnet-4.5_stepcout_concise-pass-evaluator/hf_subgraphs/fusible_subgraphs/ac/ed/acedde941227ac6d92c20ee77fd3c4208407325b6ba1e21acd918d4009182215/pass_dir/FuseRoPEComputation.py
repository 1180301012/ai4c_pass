import torch
import triton
import triton.language as tl

def pattern(in_2):
    """Pattern to match RoPE computation: cat -> cos/sin -> mul -> to(float16)"""
    tmp_2 = torch.cat((in_2, in_2), dim=-1)
    tmp_3 = tmp_2.cos()
    tmp_4 = tmp_3 * 1.0
    tmp_7 = tmp_4.to(dtype=torch.float16)
    tmp_5 = tmp_2.sin()
    tmp_6 = tmp_5 * 1.0
    tmp_8 = tmp_6.to(dtype=torch.float16)
    return tmp_7, tmp_8

def replacement_args(in_2):
    return (in_2,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_rope_kernel(
    input_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RoPE kernel: performs cat, cos, sin, mul, and dtype conversion"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input (we'll read each element twice for the cat operation)
    # The output has 2x the last dimension, so we need to figure out which input element to read
    input_size = n_elements // 2
    input_offsets = offsets % input_size
    
    # Load from input - cast to float32 for computation
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Compute cos and sin
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    # Multiply by 1.0 (already done implicitly)
    # Convert to float16 and store
    tl.store(cos_out_ptr + offsets, cos_val.to(tl.float16), mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val.to(tl.float16), mask=mask)

@torch.fx.wrap
def fused_rope(in_2):
    """Optimized RoPE computation"""
    # Input shape: [batch, seq_len, hidden_dim]
    # Output shape: [batch, seq_len, hidden_dim * 2]
    
    original_shape = in_2.shape
    input_numel = in_2.numel()
    output_numel = input_numel * 2
    
    # Allocate output tensors
    cos_out = torch.empty(original_shape[:-1] + (original_shape[-1] * 2,), 
                          dtype=torch.float16, device=in_2.device)
    sin_out = torch.empty_like(cos_out)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(output_numel, meta['BLOCK_SIZE']),)
    
    fused_rope_kernel[grid](
        in_2,
        cos_out,
        sin_out,
        output_numel,
    )
    
    return cos_out, sin_out

def replacement_func():
    return fused_rope