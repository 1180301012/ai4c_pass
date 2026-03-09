import torch
import triton
import triton.language as tl

def pattern(x):
    return x.cos()

@triton.jit
def fused_trigonometric_kernel(
    x_ptr,
    out_cos_ptr,
    out_sin_ptr,
    batch_size,
    seq_len,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // (seq_len * head_dim)
    seq_id = (pid % (seq_len * head_dim)) // head_dim
    head_id = pid % head_dim
    
    # For the doubled dimension, we need to handle both halves
    offsets_base = batch_id * seq_len * head_dim * 2 + seq_id * head_dim * 2
    
    # Process two elements at a time (original + concatenated)
    for i in range(0, head_dim, BLOCK_SIZE // 2):
        offsets = offsets_base + i * 2 + tl.arange(0, BLOCK_SIZE // 2)
        mask = i + tl.arange(0, BLOCK_SIZE // 2) < head_dim
        
        # Load original values
        x = tl.load(x_ptr + offsets // 2 + tl.arange(0, BLOCK_SIZE // 2) * head_dim, 
                   mask=mask, other=0.0)
        
        # Compute cos and sin
        cos_val = tl.cos(x)
        sin_val = tl.sin(x)
        
        # Store results in both halves of the output
        tl.store(out_cos_ptr + offsets, cos_val, mask=mask)
        tl.store(out_sin_ptr + offsets, sin_val, mask=mask)

@torch.fx.wrap
def fused_trigonometric_ops(in_1):
    batch_size, seq_len, head_dim = in_1.shape
    out_shape = (batch_size, seq_len, head_dim * 2)
    
    out_cos = torch.empty(out_shape, dtype=torch.float32, device=in_1.device)
    out_sin = torch.empty(out_shape, dtype=torch.float32, device=in_1.device)
    
    BLOCK_SIZE = 1024
    n_elements = batch_size * seq_len * head_dim * 2
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_trigonometric_kernel[(num_programs,)](
        in_1,
        out_cos,
        out_sin,
        batch_size,
        seq_len,
        head_dim,
        BLOCK_SIZE
    )
    
    return out_cos, out_sin

def replacement_args(x):
    return (x,)

@triton.jit
def simple_cos_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cosine
    out = tl.cos(x)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_cos(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    simple_cos_kernel[(num_programs,)](
        x,
        out,
        n_elements,
        BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return simple_cos