import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern to match scaled and then softmax operation"""
    tmp_0 = 0.0625 * x
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    return tmp_1

def replacement_args(x):
    """Extract arguments for replacement"""
    return (x,)

@triton.jit
def fused_scaled_softmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    seq_len,
    num_heads,
    scale_factor: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel that applies scaling followed by softmax"""
    pid = tl.program_id(0)
    
    # Calculate batch and head indices
    batch = pid // num_heads
    head = pid % num_heads
    
    if batch >= batch_size:
        return
        
    # Create offsets for the current batch and head
    batch_offset = batch * seq_len * num_heads
    head_offset = head * seq_len
    base_offset = batch_offset + head_offset
    
    # Load data for this batch/head (transpose to [batch, seq_len, num_heads])
    x = tl.load(x_ptr + base_offset + tl.arange(0, seq_len), 
                mask=tl.arange(0, seq_len) < seq_len, 
                other=tl.min_float32)
    
    # Apply scaling and then softmax
    scaled = x * scale_factor
    max_val = tl.max(scaled, 0)
    scaled = scaled - max_val
    exp_val = tl.exp(scaled)
    sum_val = tl.sum(exp_val, 0)
    softmax_out = exp_val / sum_val
    
    # Store result
    tl.store(out_ptr + base_offset + tl.arange(0, seq_len), 
             softmax_out, 
             mask=tl.arange(0, seq_len) < seq_len)

@torch.fx.wrap
def fused_scaled_softmax(x):
    """Apply fused scaled softmax operation"""
    batch, seq_len, num_heads = x.shape
    
    # Use appropriate block sizes for this tensor shape
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    
    num_programs = batch * num_heads
    
    out = torch.empty_like(x)
    
    fused_scaled_softmax_kernel[num_programs](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        scale_factor=0.0625,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    """Return the fused function"""
    return fused_scaled_softmax