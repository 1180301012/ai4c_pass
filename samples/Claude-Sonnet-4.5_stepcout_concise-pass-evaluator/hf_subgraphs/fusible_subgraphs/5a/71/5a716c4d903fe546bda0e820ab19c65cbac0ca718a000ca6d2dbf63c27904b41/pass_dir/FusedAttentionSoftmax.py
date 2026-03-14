import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match the attention score computation pattern - fused adds, div, and add"""
    tmp_0 = in_0 + in_3
    tmp_1 = tmp_0 + in_2
    tmp_2 = tmp_1 / 8.0
    tmp_3 = tmp_2 + in_1
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_add_div_add_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse: (in_0 + in_3 + in_2) / 8.0 + in_1"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    v0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    v1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    v2 = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    v3 = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation
    result = (v0 + v3 + v2) / 8.0 + v1
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_add_div_add(in_0, in_1, in_2, in_3):
    """Wrapper function to launch the fused kernel"""
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_add_div_add_kernel[grid](
        in_0, in_1, in_2, in_3, out,
        n_elements,
    )
    
    return out

def replacement_func():
    return fused_add_div_add

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['seq_len_last'],
)
@triton.jit
def fused_attention_softmax_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, out_ptr,
    batch, heads, seq_len, seq_len_last,
    stride_b0, stride_h0, stride_s0, stride_sl0,
    stride_b1, stride_h1, stride_s1, stride_sl1,
    stride_b2, stride_h2, stride_s2, stride_sl2,
    stride_b3, stride_h3, stride_s3, stride_sl3,
    stride_bo, stride_ho, stride_so, stride_slo,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes:
    - Elementwise operations: (in_0 + in_3 + in_2) / 8.0 + in_1
    - Softmax over the last dimension
    
    Each program handles one row (reduction over last dimension)
    """
    pid = tl.program_id(0)
    
    # Calculate indices for this row
    n_rows = batch * heads * seq_len
    if pid >= n_rows:
        return
    
    batch_idx = pid // (heads * seq_len)
    rem = pid % (heads * seq_len)
    head_idx = rem // seq_len
    seq_idx = rem % seq_len
    
    # Compute base pointers for each input
    base_0 = in_0_ptr + batch_idx * stride_b0 + head_idx * stride_h0 + seq_idx * stride_s0
    base_1 = in_1_ptr + batch_idx * stride_b1
    base_2 = in_2_ptr + batch_idx * stride_b2 + head_idx * stride_h2 + seq_idx * stride_s2
    base_3 = in_3_ptr + batch_idx * stride_b3 + head_idx * stride_h3 + seq_idx * stride_s3
    base_out = out_ptr + batch_idx * stride_bo + head_idx * stride_ho + seq_idx * stride_so
    
    # First pass: compute maximum for numerical stability
    max_val = -float('inf')
    for block_start in range(0, seq_len_last, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len_last
        
        v0 = tl.load(base_0 + offsets * stride_sl0, mask=mask, other=-float('inf'))
        v1 = tl.load(base_1 + offsets * stride_sl1, mask=mask, other=0.0)
        v2 = tl.load(base_2 + offsets * stride_sl2, mask=mask, other=0.0)
        v3 = tl.load(base_3 + offsets * stride_sl3, mask=mask, other=0.0)
        
        # Compute fused operation: (v0 + v3 + v2) / 8.0 + v1
        val = (v0 + v3 + v2) / 8.0 + v1
        block_max = tl.max(val)
        max_val = tl.maximum(max_val, block_max)
    
    # Second pass: compute sum of exponentials
    sum_exp = 0.0
    for block_start in range(0, seq_len_last, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len_last
        
        v0 = tl.load(base_0 + offsets * stride_sl0, mask=mask, other=0.0)
        v1 = tl.load(base_1 + offsets * stride_sl1, mask=mask, other=0.0)
        v2 = tl.load(base_2 + offsets * stride_sl2, mask=mask, other=0.0)
        v3 = tl.load(base_3 + offsets * stride_sl3, mask=mask, other=0.0)
        
        val = (v0 + v3 + v2) / 8.0 + v1
        exp_val = tl.exp(val - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_val, 0.0))
    
    # Third pass: compute softmax values and store
    for block_start in range(0, seq_len_last, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < seq_len_last
        
        v0 = tl.load(base_0 + offsets * stride_sl0, mask=mask, other=0.0)
        v1 = tl.load(base_1 + offsets * stride_sl1, mask=mask, other=0.0)
        v2 = tl.load(base_2 + offsets * stride_sl2, mask=mask, other=0.0)
        v3 = tl.load(base_3 + offsets * stride_sl3, mask=mask, other=0.0)
        
        val = (v0 + v3 + v2) / 8.0 + v1
        softmax_val = tl.exp(val - max_val) / sum_exp
        
        tl.store(base_out + offsets * stride_slo, softmax_val, mask=mask)

@torch.fx.wrap
def fused_attention_softmax(in_0, in_1, in_2, in_3):
    """Wrapper function to launch the fused kernel"""
    # Get shape information
    batch, heads, seq_len, seq_len_last = in_0.shape
    
    # Allocate output
    out = torch.empty_like(in_0)
    
    # Launch kernel - one program per row
    grid = (batch * heads * seq_len,)
    
    fused_attention_softmax_kernel[grid](
        in_0, in_1, in_2, in_3, out,
        batch, heads, seq_len, seq_len_last,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    
    return out

def replacement_func():
    return fused_attention_softmax