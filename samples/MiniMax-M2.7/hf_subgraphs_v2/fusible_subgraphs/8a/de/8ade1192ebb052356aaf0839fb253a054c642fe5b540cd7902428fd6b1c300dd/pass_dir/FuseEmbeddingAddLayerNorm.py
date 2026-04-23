import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match: in_0 + embedding(unsqueeze(in_4) + 2, in_1) -> layer_norm -> dropout
    Pattern matches the core computation path:
    - unsqueeze positions (in_4)
    - add offset
    - embedding lookup (in_1)
    - move to cuda
    - add with input (in_0)
    - layer_norm (in_3=weight, in_2=bias)
    - dropout (disabled)
    """
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), in_3, in_2, 1e-05)
    tmp_15 = torch.nn.functional.dropout(tmp_14, p=0.1, training=False)
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_kernel_1024(
    positions_ptr, embed_table_ptr, input_ptr, 
    weight_ptr, bias_ptr, output_ptr,
    seq_len: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes one output element
    pid = tl.program_id(0)
    pos = pid // 1024
    dim = pid % 1024
    
    if pos >= seq_len:
        return
    
    # Load position and compute embedding
    pos_val = tl.load(positions_ptr + pos)
    embed_offset = pos_val * 1024 + dim
    embed_val = tl.load(embed_table_ptr + embed_offset)
    
    # Load input and add
    input_offset = pos * 1024 + dim
    input_val = tl.load(input_ptr + input_offset)
    combined = input_val + embed_val
    
    # Layer norm (fused reduction across hidden dim)
    row_base = pos * 1024
    
    # Load full row for reduction
    row_offsets = row_base + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < (pos + 1) * 1024
    row_vals = tl.load(row_base + row_offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(row_vals) / 1024.0
    
    # Compute variance
    diff = row_vals - mean
    diff_sq = diff * diff
    var = tl.sum(diff_sq) / 1024.0
    
    # Normalize
    normalized = (combined - mean) * tl.rsqrt(var + 1e-05)
    
    # Apply weight and bias
    w = tl.load(weight_ptr + dim)
    b = tl.load(bias_ptr + dim)
    result = normalized * w + b
    
    # Store
    tl.store(output_ptr + input_offset, result)


@triton.jit
def fused_kernel_768(
    positions_ptr, embed_table_ptr, input_ptr, 
    weight_ptr, bias_ptr, output_ptr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = pid // 768
    dim = pid % 768
    
    if pos >= seq_len:
        return
    
    pos_val = tl.load(positions_ptr + pos)
    embed_offset = pos_val * 768 + dim
    embed_val = tl.load(embed_table_ptr + embed_offset)
    
    input_offset = pos * 768 + dim
    input_val = tl.load(input_ptr + input_offset)
    combined = input_val + embed_val
    
    row_base = pos * 768
    row_offsets = row_base + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < (pos + 1) * 768
    row_vals = tl.load(row_base + row_offsets, mask=mask, other=0.0)
    
    mean = tl.sum(row_vals) / 768.0
    diff = row_vals - mean
    diff_sq = diff * diff
    var = tl.sum(diff_sq) / 768.0
    
    normalized = (combined - mean) * tl.rsqrt(var + 1e-05)
    w = tl.load(weight_ptr + dim)
    b = tl.load(bias_ptr + dim)
    result = normalized * w + b
    
    tl.store(output_ptr + input_offset, result)


@triton.jit
def fused_kernel_16(
    positions_ptr, embed_table_ptr, input_ptr, 
    weight_ptr, bias_ptr, output_ptr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = pid // 16
    dim = pid % 16
    
    if pos >= seq_len:
        return
    
    pos_val = tl.load(positions_ptr + pos)
    embed_offset = pos_val * 16 + dim
    embed_val = tl.load(embed_table_ptr + embed_offset)
    
    input_offset = pos * 16 + dim
    input_val = tl.load(input_ptr + input_offset)
    combined = input_val + embed_val
    
    row_base = pos * 16
    row_offsets = row_base + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < (pos + 1) * 16
    row_vals = tl.load(row_base + row_offsets, mask=mask, other=0.0)
    
    mean = tl.sum(row_vals) / 16.0
    diff = row_vals - mean
    diff_sq = diff * diff
    var = tl.sum(diff_sq) / 16.0
    
    normalized = (combined - mean) * tl.rsqrt(var + 1e-05)
    w = tl.load(weight_ptr + dim)
    b = tl.load(bias_ptr + dim)
    result = normalized * w + b
    
    tl.store(output_ptr + input_offset, result)


@torch.fx.wrap
def fused_embedding_add_layer_norm_impl(input_tensor, embed_table, bias, weight, positions):
    """
    Fused: unsqueeze + add + embedding + to(cuda) + add + layer_norm + dropout(disabled)
    
    Dropout is disabled (p=0.1, training=False), so we skip it.
    """
    # Compute positions: unsqueeze + add offset
    positions_2d = positions.unsqueeze(0) + 2
    
    seq_len = positions_2d.shape[1]
    hidden_dim = input_tensor.shape[2]
    embed_dim = embed_table.shape[1]
    
    # Allocate output
    output = torch.empty_like(input_tensor)
    
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    
    if hidden_dim == 1024:
        grid = (seq_len * 1024,)
        fused_kernel_1024[grid](
            positions_2d, embed_table, input_tensor, weight, bias, output,
            seq_len, BLOCK_SIZE
        )
    elif hidden_dim == 768:
        grid = (seq_len * 768,)
        fused_kernel_768[grid](
            positions_2d, embed_table, input_tensor, weight, bias, output,
            seq_len, BLOCK_SIZE
        )
    elif hidden_dim == 16:
        grid = (seq_len * 16,)
        fused_kernel_16[grid](
            positions_2d, embed_table, input_tensor, weight, bias, output,
            seq_len, BLOCK_SIZE
        )
    else:
        # Fallback to original computation for other hidden dims
        positions_2d_cuda = positions_2d.to(device='cuda')
        embedded = torch.nn.functional.embedding(positions_2d_cuda, embed_table, None, None, 2.0, False, False)
        combined = input_tensor + embedded
        output = torch.nn.functional.layer_norm(combined, (hidden_dim,), weight, bias, 1e-05)
    
    return output


def replacement_func():
    return fused_embedding_add_layer_norm_impl