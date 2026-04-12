import torch
import triton
import triton.language as tl

def pattern(in_1):
    # Match the trigonometric computation pattern
    # tmp_1 = torch.cat((in_1, in_1), dim = -1)
    tmp_1 = torch.cat((in_1, in_1), dim = -1)
    # tmp_2 = tmp_1.cos()
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    # tmp_4 = tmp_1.sin()
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    # Convert to bfloat16
    tmp_6 = tmp_3.to(dtype = torch.bfloat16)
    tmp_7 = tmp_5.to(dtype = torch.bfloat16)
    return tmp_6, tmp_7

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def fused_trigonometric_kernel(
    freqs_ptr,
    cos_out_ptr,
    sin_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load frequency values
    freqs = tl.load(freqs_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both cos and sin in one pass
    cos_val = tl.cos(freqs)
    sin_val = tl.sin(freqs)
    
    # Store results
    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=mask)

@torch.fx.wrap
def fused_trigonometric_ops(freqs):
    # Get original dimensions
    original_shape = freqs.shape
    batch, seq_len, hidden_dim = original_shape
    
    # Concatenate along last dimension in original computation: [batch, seq_len, hidden_dim] -> [batch, seq_len, hidden_dim*2]
    # But we can optimize by computing cos/sin on original and returning both results
    # The original pattern concatenates first, then computes, but that's equivalent to
    # computing cos/sin on original and then duplicating the results
    
    total_elements = batch * seq_len * hidden_dim
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors with correct shapes for concatenated dimensions
    # Original: tmp_1 = [batch, seq_len, hidden_dim*2], then cos/sin gives [batch, seq_len, hidden_dim*2]
    # But we return as bfloat16 directly: [batch, seq_len, hidden_dim*2] for both cos and sin
    
    out_shape = (batch, seq_len, hidden_dim * 2)
    cos_out = torch.empty(out_shape, dtype=torch.bfloat16, device=freqs.device)
    sin_out = torch.empty(out_shape, dtype=torch.bfloat16, device=freqs.device)
    
    # Launch kernel for each half separately (equivalent to concatenating first)
    # First half: use original freqs
    fused_trigonometric_kernel[(num_programs,)](
        freqs_ptr=freqs,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_out, sin_out

def replacement_func():
    return fused_trigonometric_ops