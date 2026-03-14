import torch
import triton
import triton.language as tl


def pattern(in_2, tmp_7, in_1, in_0):
    """Simple pattern: just add + layernorm"""
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), in_1, in_0, 1e-05)
    return (tmp_8, tmp_9)


def replacement_args(in_2, tmp_7, in_1, in_0):
    return (in_2, tmp_7, in_1, in_0)


@triton.jit
def add_layernorm_kernel_96(
    in_2_ptr,
    tmp_7_ptr,
    weight_ptr,
    bias_ptr,
    out_add_ptr,
    out_ln_ptr,
    n_elements,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + layernorm kernel"""
    row_idx = tl.program_id(0)
    
    # Calculate offsets
    row_start = row_idx * hidden_dim
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim
    
    # Load inputs
    in_2_val = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    tmp_7_val = tl.load(tmp_7_ptr + offsets, mask=mask, other=0.0)
    
    # Add
    add_result = in_2_val + tmp_7_val
    
    # Store add result
    tl.store(out_add_ptr + offsets, add_result, mask=mask)
    
    # LayerNorm
    mean = tl.sum(add_result) / hidden_dim
    centered = add_result - mean
    var = tl.sum(centered * centered) / hidden_dim
    rstd = 1.0 / tl.sqrt(var + 1e-05)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # Normalize
    ln_result = centered * rstd * weight + bias
    
    # Store layernorm result
    tl.store(out_ln_ptr + offsets, ln_result, mask=mask)


@torch.fx.wrap
def add_layernorm_impl_96(in_2, tmp_7, in_1, in_0):
    """Fused add + layernorm implementation"""
    batch_size = in_2.shape[0]
    seq_len = in_2.shape[1]
    hidden_dim = in_2.shape[2]
    
    # Allocate outputs
    out_add = torch.empty_like(in_2)
    out_ln = torch.empty_like(in_2)
    
    # Launch kernel
    BLOCK_SIZE = 128
    n_rows = batch_size * seq_len
    grid = (n_rows,)
    
    add_layernorm_kernel_96[grid](
        in_2,
        tmp_7,
        in_1,
        in_0,
        out_add,
        out_ln,
        in_2.numel(),
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_add, out_ln)


def replacement_func():
    return add_layernorm_impl_96