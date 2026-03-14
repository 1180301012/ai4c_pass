import torch
import triton
import triton.language as tl


@triton.jit
def fused_rope_v2_kernel(
    neg_ptr,       # negate input (shape: [1, num_heads, seq_len, 32])
    slice_ptr,     # slice input (shape: [1, num_heads, seq_len, 64])
    add_ptr,       # add input (shape: [1, num_heads, seq_len, 64])
    mul_ptr,       # mul input (shape: [seq_len, 64])
    out_ptr,       # output
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim_half: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one (head, seq) position
    # Grid: (num_heads, seq_len)
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    # Calculate offsets
    # neg input: [1, num_heads, seq_len, 32]
    # slice input: [1, num_heads, seq_len, 64]
    # add input: [1, num_heads, seq_len, 64]
    # mul input: [seq_len, 64]
    
    # offset for neg input
    offset_neg = pid_h * seq_len * head_dim_half * 2 + pid_s * head_dim_half * 2
    
    # offset for slice input
    offset_slice = pid_h * seq_len * 64 + pid_s * 64
    
    # offset for add input
    offset_add = pid_h * seq_len * 64 + pid_s * 64
    
    # offset for output
    offset_out = pid_h * seq_len * 64 + pid_s * 64
    
    # Load negated values from neg input (32 values)
    neg_vals = tl.load(neg_ptr + offset_neg + tl.arange(0, 32)).to(tl.float32)
    neg_vals = -neg_vals  # Negate
    
    # Load slice input values with stride 2 (every other element = 32 values from 64)
    slice_offsets = offset_slice + tl.arange(0, 32) * 2
    slice_vals = tl.load(slice_ptr + slice_offsets).to(tl.float32)
    
    # Interleave: even positions get negated values, odd positions get sliced values
    idx = tl.arange(0, 32)
    interleaved = tl.where(idx % 2 == 0, neg_vals, slice_vals)
    
    # Load mul input (sin_emb) - shape [seq_len, 64], same for all heads
    mul_vals = tl.load(mul_ptr + pid_s * 64 + tl.arange(0, 64)).to(tl.float32)
    
    # Multiply
    mul_result = interleaved * mul_vals
    
    # Load add input
    add_vals = tl.load(add_ptr + offset_add + tl.arange(0, 64)).to(tl.float32)
    
    # Add
    final_result = add_vals + mul_result
    
    # Store output
    tl.store(out_ptr + offset_out + tl.arange(0, 64), final_result)


@torch.fx.wrap
def fused_rope_v2_wrapper(neg_input, slice_input, add_input, mul_input):
    """
    Fused kernel for RoPE computation (Variant 2):
    - tmp_1 = -neg_input
    - tmp_2 = slice_input[Ellipsis, ::2]
    - tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    - tmp_4 = tmp_3.reshape((1, num_heads, seq_len, 64))
    - tmp_5 = tmp_4 * mul_input
    - tmp_6 = add_input + tmp_5
    
    Input shapes:
    - neg_input: [1, num_heads, seq_len, 32]
    - slice_input: [1, num_heads, seq_len, 64]
    - add_input: [1, num_heads, seq_len, 64]
    - mul_input: [seq_len, 64]
    
    Output shape:
    - [1, num_heads, seq_len, 64]
    """
    batch, num_heads, seq_len, _ = neg_input.shape
    _, _, _, head_dim64 = add_input.shape
    
    # Allocate output
    out = torch.empty_like(add_input)
    
    # Launch kernel
    grid = (num_heads, seq_len)
    
    fused_rope_v2_kernel[grid](
        neg_input,
        slice_input,
        add_input,
        mul_input,
        out,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim_half=32,
        BLOCK_SIZE=1024,
    )
    
    return out


# Pattern for Models 2-4: (-in_3, in_2[::2], in_5, in_6)
# Using generic argument names
def pattern(a, b, c, d):
    """
    Pattern to match (Model 2-4 variant):
    tmp_1 = -a
    tmp_2 = b[Ellipsis, slice(None, None, 2)]
    tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    tmp_4 = tmp_3.reshape((1, num_heads, seq_len, 64))
    tmp_5 = tmp_4 * d
    tmp_6 = c + tmp_5
    return tmp_6
    """
    tmp_1 = -a
    tmp_2 = b[..., slice(None, None, 2)]
    tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    tmp_4 = tmp_3.reshape((1, tmp_3.shape[1], tmp_3.shape[2], 64))
    tmp_5 = tmp_4 * d
    tmp_6 = c + tmp_5
    return tmp_6


def replacement_args(a, b, c, d):
    return (a, b, c, d)


def replacement_func():
    return fused_rope_v2_wrapper