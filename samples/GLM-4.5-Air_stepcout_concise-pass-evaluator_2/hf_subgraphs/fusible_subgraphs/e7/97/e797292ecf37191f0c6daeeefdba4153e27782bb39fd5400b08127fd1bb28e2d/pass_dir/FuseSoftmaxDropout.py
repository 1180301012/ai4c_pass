import torch
import triton
import triton.language as tl
import math

@triton.jit
def triton_softmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    features,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch and sequence
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return
    
    # Compute base offset for this batch and sequence
    base_offset = batch_idx * seq_len * features + seq_idx * features
    
    # Load slice for softmax computation
    offsets = base_offset + tl.arange(0, features)
    x_slice = tl.load(x_ptr + offsets, mask=offsets < (batch_idx * seq_len * features + (seq_idx + 1) * features), other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x_slice)
    
    # Compute exponentials
    x_exp = tl.exp(x_slice - max_val)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(x_exp)
    
    # Compute softmax
    softmax = x_exp / sum_exp
    
    # Store output
    tl.store(out_ptr + offsets, softmax, mask=offsets < (batch_idx * seq_len * features + (seq_idx + 1) * features))

@torch.fx.wrap
def triton_softmax(x):
    """Optimized softmax implementation that handles dropout with p=0.0"""
    # Since dropout p=0.0 is identity, we just need to compute softmax
    
    # Handle different input shapes - softmax on last dimension
    if x.dim() == 3:
        # Shape is [batch_size, seq_len, features] - softmax on last dim
        batch_size, seq_len, features = x.shape
        out = torch.empty_like(x)
        
        BLOCK_SIZE = 1024  # Number of programs launched (we'll handle per batch-sequence)
        num_programs = batch_size * seq_len
        
        triton_softmax_kernel[(num_programs, 1, 1)](
            x_ptr=x,
            out_ptr=out,
            batch_size=batch_size,
            seq_len=seq_len,
            features=features,
            BLOCK_SIZE=features,
        )
        
        return out
    else:
        # Fallback to torch softmax for other shapes
        return torch.nn.functional.softmax(x, dim=-1)

def pattern(attn_weights):
    """Pattern: softmax followed by dropout with p=0.0"""
    softmax_out = torch.nn.functional.softmax(attn_weights, dim=-1)
    dropout_out = torch.nn.functional.dropout(softmax_out, p=0.0, training=False)
    return dropout_out

def replacement_args(attn_weights):
    return (attn_weights,)

def replacement_func():
    return triton_softmax