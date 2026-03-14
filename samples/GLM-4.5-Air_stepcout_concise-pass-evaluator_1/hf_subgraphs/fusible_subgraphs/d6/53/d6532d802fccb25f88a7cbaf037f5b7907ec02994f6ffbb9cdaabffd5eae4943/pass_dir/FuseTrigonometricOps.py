import torch
import triton
import triton.language as tl

def pattern(freqs):
    # The pattern should exactly mirror the operations from the original model
    tmp_1 = torch.cat((freqs, freqs), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    # Return the values that are observable outside the matched subgraph
    return tmp_6, tmp_7

def replacement_args(freqs):
    return (freqs,)

@triton.jit
def fused_trigonometric_kernel(
    freqs_ptr,
    cos_out_ptr,
    sin_out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one batch, seq_len position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Compute offset in the frequency tensor (dim=64)
    freq_offset = batch_idx * seq_len * 64 + seq_idx * 64
    
    # Load frequency values (64-dim)
    freqs = tl.load(freqs_ptr + freq_offset + tl.arange(0, 64), mask=tl.arange(0, 64) < 64, other=0.0)
    
    # Compute cos and sin (both will be bfloat16)
    cos_vals = tl.cos(freqs).to(tl.bfloat16)
    sin_vals = tl.sin(freqs).to(tl.bfloat16)
    
    # Store results duplicated for both halves (cos and sin each occupy full 128 dim)
    cos_out_offset = batch_idx * seq_len * 128 + seq_idx * 128
    sin_out_offset = batch_idx * seq_len * 128 + seq_idx * 128
    
    # Store cos values duplicated (first 64 and second 64 are same)
    tl.store(cos_out_ptr + cos_out_offset + tl.arange(0, 64), cos_vals, mask=tl.arange(0, 64) < 64)
    tl.store(cos_out_ptr + cos_out_offset + tl.arange(64, 128), cos_vals, mask=tl.arange(64, 128) < 64)
    
    # Store sin values duplicated (first 64 and second 64 are same)  
    tl.store(sin_out_ptr + sin_out_offset + tl.arange(0, 64), sin_vals, mask=tl.arange(0, 64) < 64)
    tl.store(sin_out_ptr + sin_out_offset + tl.arange(64, 128), sin_vals, mask=tl.arange(64, 128) < 64)

@torch.fx.wrap  
def fused_trigonometric_ops(freqs):
    # Get input shape
    batch_size, seq_len, dim = freqs.shape
    assert dim == 64, "Expected dim=64 for freqs"
    
    # Create output tensors
    cos_out = torch.empty((batch_size, seq_len, 128), dtype=torch.bfloat16, device=freqs.device)
    sin_out = torch.empty((batch_size, seq_len, 128), dtype=torch.bfloat16, device=freqs.device)
    
    # Set up launch grid - one program per (batch, seq_len) pair
    grid = (batch_size, seq_len)
    
    fused_trigonometric_kernel[grid](
        freqs,
        cos_out,
        sin_out,
        batch_size,
        seq_len,
        BLOCK_SIZE_N=64,
    )
    
    return cos_out, sin_out

def replacement_func():
    # Start with a simple non-fused version first to test pattern matching
    def simple_trigonometric_ops(freqs):
        # This reproduces the original operations exactly without optimization
        tmp_1 = torch.cat((freqs, freqs), dim=-1)
        tmp_2 = tmp_1.cos()
        tmp_3 = tmp_2 * 1.0
        tmp_4 = tmp_1.sin()
        tmp_5 = tmp_4 * 1.0
        tmp_6 = tmp_3.to(dtype=torch.bfloat16)
        tmp_7 = tmp_5.to(dtype=torch.bfloat16)
        return tmp_6, tmp_7
    
    return simple_trigonometric_ops