import torch
import triton
import triton.language as tl


@triton.jit
def value_reshape_kernel(
    value_ptr,
    output_ptr,
    stride_v_batch, stride_v_head, stride_v_seq, stride_v_dim,
    stride_o_batch, stride_o_head, stride_o_seq, stride_o_dim,
    n_batch, n_seq, n_dim,
    num_heads,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused reshape and expand for value states.
    Input: [batch, 1, seq, dim] 
    Output: [batch, num_heads, seq, dim]
    The kernel broadcasts from head_dim=1 to num_heads.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Compute output offsets
    out_offset_base = (
        batch_idx * stride_o_batch +
        head_idx * stride_o_head +
        seq_idx * stride_o_seq
    )
    
    # Compute input offsets (value has head_dim=1)
    in_offset_base = (
        batch_idx * stride_v_batch +
        0 * stride_v_head +
        seq_idx * stride_v_seq
    )
    
    # Load all values and store (broadcasting along head dimension)
    for dim_idx in range(0, n_dim, BLOCK_SIZE):
        offsets = dim_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_dim
        
        val = tl.load(value_ptr + in_offset_base + offsets * stride_v_dim, mask=mask, other=0.0)
        tl.store(output_ptr + out_offset_base + offsets, val, mask=mask)


@torch.fx.wrap
def fused_value_reshape(value_states, num_heads=8):
    """
    Fused reshape for value_states.
    Input shape: [1, 1, 3, 256]
    Output shape: [1, 8, 3, 256]
    """
    batch, head_in, seq, dim = value_states.shape
    n_head_out = num_heads
    
    # Allocate output
    output = torch.empty(batch, n_head_out, seq, dim, dtype=value_states.dtype, device=value_states.device)
    
    # Grid configuration
    grid = (batch, n_head_out, seq)
    BLOCK_SIZE = 256
    
    # Run kernel
    value_reshape_kernel[grid](
        value_states, output,
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch, seq, dim, n_head_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def pattern(in_5):
    """
    Match the value_states reshape pattern.
    Operations matched:
    - tmp_10 = in_5[..., None]
    - tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    - tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    
    Returns tmp_12 (the reshaped output).
    """
    tmp_10 = in_5[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    return tmp_12


def replacement_args(in_5):
    return (in_5,)


def replacement_func():
    return fused_value_reshape