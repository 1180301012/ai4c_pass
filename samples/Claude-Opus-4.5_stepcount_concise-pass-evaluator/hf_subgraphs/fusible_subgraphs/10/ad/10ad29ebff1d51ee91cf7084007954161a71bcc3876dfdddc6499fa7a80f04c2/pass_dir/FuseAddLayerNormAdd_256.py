import torch
import triton
import triton.language as tl
import operator

# Pattern matching function - must match exactly the operations in model.py
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (256,), in_1, in_0, 1e-05)
    tmp_4 = tmp_3 + in_4
    return tmp_3, tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

@triton.jit
def fused_add_layernorm_add_kernel(
    in_2_ptr, in_3_ptr, in_4_ptr,
    weight_ptr, bias_ptr,
    out_tmp3_ptr, out_tmp4_ptr,
    num_rows,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Thread index within block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    # Compute offsets for this row
    row_start = row_idx * hidden_size
    
    # Load data for this row
    in_2 = tl.load(in_2_ptr + row_start + offsets, mask=mask, other=0.0)
    in_3 = tl.load(in_3_ptr + row_start + offsets, mask=mask, other=0.0)
    
    # First add
    x = in_2 + in_3
    
    # Layer norm computation
    mean = tl.sum(x, axis=0) / hidden_size
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    rstd = tl.rsqrt(var + eps)
    x_norm = x_centered * rstd
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Apply affine transformation
    tmp_3 = x_norm * weight + bias
    
    # Load in_4 and compute tmp_4
    in_4 = tl.load(in_4_ptr + row_start + offsets, mask=mask, other=0.0)
    tmp_4 = tmp_3 + in_4
    
    # Store results
    tl.store(out_tmp3_ptr + row_start + offsets, tmp_3, mask=mask)
    tl.store(out_tmp4_ptr + row_start + offsets, tmp_4, mask=mask)

@torch.fx.wrap
def fused_add_layernorm_add_impl(in_0, in_1, in_2, in_3, in_4):
    # in_0: bias [256]
    # in_1: weight [256]
    # in_2, in_3, in_4: [1, 100, 256]
    
    batch_size = in_2.shape[0]
    seq_len = in_2.shape[1]
    hidden_size = in_2.shape[2]
    
    num_rows = batch_size * seq_len
    
    out_tmp3 = torch.empty_like(in_2)
    out_tmp4 = torch.empty_like(in_2)
    
    fused_add_layernorm_add_kernel[(num_rows,)](
        in_2, in_3, in_4,
        in_1, in_0,  # weight, bias
        out_tmp3, out_tmp4,
        num_rows,
        hidden_size,
        1e-05,  # eps
        BLOCK_SIZE=256,
        num_warps=1,
    )
    
    return (out_tmp3, out_tmp4)

# Wrapper that extracts tuple elements to create separate return nodes
def fused_add_layernorm_add(in_0, in_1, in_2, in_3, in_4):
    result = fused_add_layernorm_add_impl(in_0, in_1, in_2, in_3, in_4)
    return operator.getitem(result, 0), operator.getitem(result, 1)

# Replacement function - returns the callable
def replacement_func():
    return fused_add_layernorm_add