import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """Pattern for Graph 3: 35x35x384 -> 32x32x384"""
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 35, 35, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None)]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 1024, 384)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), in_1, in_0, 1e-05)
    return (tmp_8, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def fused_kernel_384(
    in_3_ptr,
    in_2_ptr,
    weight_ptr,
    bias_ptr,
    out_add_ptr,
    out_ln_ptr,
    seq_len,
    HIDDEN_DIM: tl.constexpr,
    H_PADDED: tl.constexpr,
    W_PADDED: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for roll + slice + add + layernorm"""
    token_idx = tl.program_id(0)
    
    if token_idx >= seq_len:
        return
    
    # Calculate output position
    out_h = token_idx // W_OUT
    out_w = token_idx % W_OUT
    
    # Apply reverse roll to find source position
    src_h = (out_h - SHIFT) % H_PADDED
    src_w = (out_w - SHIFT) % W_PADDED
    
    # Load data
    hidden_offset = tl.arange(0, BLOCK_SIZE)
    mask = hidden_offset < HIDDEN_DIM
    
    # Load from rolled source
    src_offset = src_h * W_PADDED * HIDDEN_DIM + src_w * HIDDEN_DIM + hidden_offset
    rolled_val = tl.load(in_3_ptr + src_offset, mask=mask, other=0.0)
    
    # Load from in_2
    in_2_offset = token_idx * HIDDEN_DIM + hidden_offset
    in_2_val = tl.load(in_2_ptr + in_2_offset, mask=mask, other=0.0)
    
    # Add
    add_result = in_2_val + rolled_val
    
    # Store add result
    tl.store(out_add_ptr + in_2_offset, add_result, mask=mask)
    
    # LayerNorm computation
    mean = tl.sum(add_result, axis=0) / HIDDEN_DIM
    centered = add_result - mean
    var = tl.sum(centered * centered, axis=0) / HIDDEN_DIM
    rstd = 1.0 / tl.sqrt(var + 1e-05)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + hidden_offset, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + hidden_offset, mask=mask, other=0.0)
    
    # Normalize
    ln_result = centered * rstd * weight + bias
    
    # Store layernorm result
    tl.store(out_ln_ptr + in_2_offset, ln_result, mask=mask)


@torch.fx.wrap
def fused_impl_384(in_0, in_1, in_2, in_3):
    """Optimized implementation for 384-dim hidden size"""
    batch_size = in_3.shape[0]
    hidden_dim = 384
    h_padded = 35
    w_padded = 35
    h_out = 32
    w_out = 32
    seq_len = h_out * w_out
    
    # Reshape in_3 to padded view
    in_3_reshaped = in_3.contiguous().view(-1, h_padded, w_padded, hidden_dim)
    
    # Allocate outputs
    out_add = torch.empty((batch_size, seq_len, hidden_dim), device=in_2.device, dtype=in_2.dtype)
    out_ln = torch.empty_like(out_add)
    
    # Launch kernel
    BLOCK_SIZE = 512
    grid = (seq_len,)
    
    fused_kernel_384[grid](
        in_3_reshaped,
        in_2,
        in_1,
        in_0,
        out_add,
        out_ln,
        seq_len,
        HIDDEN_DIM=hidden_dim,
        H_PADDED=h_padded,
        W_PADDED=w_padded,
        H_OUT=h_out,
        W_OUT=w_out,
        SHIFT=3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out_add, out_ln)


def replacement_func():
    return fused_impl_384