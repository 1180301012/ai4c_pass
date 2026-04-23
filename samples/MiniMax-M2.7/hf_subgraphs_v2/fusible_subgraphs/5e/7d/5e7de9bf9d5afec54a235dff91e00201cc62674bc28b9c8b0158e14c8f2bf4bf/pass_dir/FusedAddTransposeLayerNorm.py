import torch
import triton
import triton.language as tl

# ============================================================================
# Fused Add + Transpose + LayerNorm Pass
# ============================================================================
# This pass fuses the add, transpose, and layer_norm operations.
# Inputs: tmp_6 [1, 768, 124], tmp_7 [1, 48, 124], in_0 [768], in_1 [768]
# Output: tmp_10 [1, 124, 768]

@triton.jit
def fused_add_transpose_layernorm_kernel(
    a_ptr, b_ptr, weight_ptr, bias_ptr, out_ptr,
    batch, dim_a, dim_b,  # dim_a=768, dim_b=48
    seq_len,  # 124
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    
    # First pass: compute sum for mean
    sum_vals = 0.0
    for ch in range(dim_a):
        a_val = tl.load(a_ptr + (batch_idx * dim_a * seq_len + ch * seq_len + seq_idx))
        if ch < dim_b:
            b_val = tl.load(b_ptr + (batch_idx * dim_b * seq_len + ch * seq_len + seq_idx))
        else:
            b_val = 0.0
        sum_vals += a_val + b_val
    
    mean = sum_vals / tl.cast(dim_a, tl.float32)
    
    # Second pass: compute variance
    sq_sum = 0.0
    for ch in range(dim_a):
        a_val = tl.load(a_ptr + (batch_idx * dim_a * seq_len + ch * seq_len + seq_idx))
        if ch < dim_b:
            b_val = tl.load(b_ptr + (batch_idx * dim_b * seq_len + ch * seq_len + seq_idx))
        else:
            b_val = 0.0
        val = a_val + b_val
        sq_sum += (val - mean) * (val - mean)
    
    variance = sq_sum / tl.cast(dim_a, tl.float32)
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Third pass: normalize and store
    for ch in range(dim_a):
        a_val = tl.load(a_ptr + (batch_idx * dim_a * seq_len + ch * seq_len + seq_idx))
        if ch < dim_b:
            b_val = tl.load(b_ptr + (batch_idx * dim_b * seq_len + ch * seq_len + seq_idx))
        else:
            b_val = 0.0
        
        val = a_val + b_val
        normalized = (val - mean) * inv_std
        
        # Affine transform: normalized * weight + bias
        w = tl.load(weight_ptr + ch)
        b = tl.load(bias_ptr + ch)
        out_val = normalized * w + b
        
        # Store output (transposed: [batch, seq, dim] -> [batch, seq, dim])
        out_idx = batch_idx * dim_a * seq_len + seq_idx * dim_a + ch
        tl.store(out_ptr + out_idx, out_val)


@torch.fx.wrap
def fused_add_transpose_layernorm(tmp_6, tmp_7, in_0, in_1):
    """Fused add + transpose + layer_norm kernel wrapper"""
    # tmp_6 shape: [1, 768, 124], tmp_7 shape: [1, 48, 124]
    batch, dim_a, seq_len = tmp_6.shape
    _, dim_b, _ = tmp_7.shape
    
    # Output shape (after transpose): [1, 124, 768]
    out = torch.empty((batch, seq_len, dim_a), 
                      dtype=tmp_6.dtype, device=tmp_6.device)
    
    total_elements = batch * seq_len
    
    # Launch kernel
    BLOCK_SIZE = 128
    num_programs = total_elements
    
    fused_add_transpose_layernorm_kernel[(num_programs,)](
        a_ptr=tmp_6, b_ptr=tmp_7, 
        weight_ptr=in_1, bias_ptr=in_0, 
        out_ptr=out,
        batch=batch, dim_a=dim_a, dim_b=dim_b,
        seq_len=seq_len,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, tmp_6, tmp_7):
    """Match: tmp_6 + tmp_7 -> transpose -> layer_norm"""
    tmp_8 = tmp_6 + tmp_7
    tmp_9 = tmp_8.transpose(1, 2)
    tmp_10 = torch.nn.functional.layer_norm(tmp_9, (768,), in_1, in_0, 1e-05)
    return tmp_10


def replacement_args(in_0, in_1, tmp_6, tmp_7):
    """Extract arguments for the fused kernel"""
    return (tmp_6, tmp_7, in_0, in_1)


def replacement_func():
    """Return the fused add+transpose+layernorm kernel"""
    return fused_add_transpose_layernorm