import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Match the computation pattern with p=0.0 dropout:
    softmax -> dropout(p=0.0) -> bmm -> reshape operations
    """
    tmp_0 = torch.nn.functional.softmax(in_0, dim=-1)
    tmp_1 = torch.nn.functional.dropout(tmp_0, p=0.0, training=False)
    tmp_2 = torch.bmm(tmp_1, in_1)
    # The reshape operations need to be included to make the pattern observable
    tmp_3 = tmp_2.view(1, tmp_2.shape[1], 1, tmp_2.shape[2])
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.reshape(1, 1, -1)
    return tmp_5

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized kernel that eliminates the dropout and potentially optimizes the attention computation
@triton.jit
def optimized_attention_kernel(
    attn_weights_ptr,
    value_states_ptr,
    output_ptr,
    batch_size,
    seq_len,
    value_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    
    # Handle each batch item
    if pid < batch_size:
        # Compute offsets for this batch item
        attn_offset = pid * seq_len  # attn_weights is [batch_size, seq_len, 1]
        value_offset = pid * seq_len * value_dim  # value_states is [batch_size, seq_len, value_dim]
        
        # Load attention weights (softmax should be done within kernel or before)
        # For now, softmax is done in the wrapper, but we can optimize this further
        attn = tl.load(attn_weights_ptr + attn_offset, mask=attn_offset < batch_size * seq_len, other=0.0)
        
        # Load value states for this batch item and all heads
        values = tl.load(value_states_ptr + value_offset, 
                        mask=value_offset < batch_size * seq_len * value_dim, 
                        other=0.0)
        
        # Softmax on attention weights (axis=-1, but since last dim=1, just exponentiate)
        max_val = tl.max(attn)
        exp_attn = tl.exp(attn - max_val)
        sum_exp = tl.sum(exp_attn)
        softmax_attn = exp_attn / (sum_exp + 1e-8)
        
        # Compute weighted sum of value states
        weighted_sum = 0.0
        for k in range(seq_len):
            if k < seq_len:  # Ensure we don't go out of bounds
                attn_val = tl.load(attn_weights_ptr + attn_offset + k, mask=(attn_offset + k) < batch_size * seq_len, other=0.0)
                # Apply softmax
                exp_attn_k = tl.exp(attn_val - max_val)
                sum_exp_k = tl.sum(exp_attn)
                softmax_attn_k = exp_attn_k / (sum_exp_k + 1e-8)
                
                # Load value for this head
                values_k = tl.load(value_states_ptr + value_offset + k * value_dim, 
                                 mask=(value_offset + k * value_dim) < batch_size * seq_len * value_dim, 
                                 other=0.0)
                
                weighted_sum += softmax_attn_k * values_k
        
        # Store the result (output should be [1, 1, batch_size * value_dim])
        output_offset = pid * value_dim
        tl.store(output_ptr + output_offset, weighted_sum, mask=output_offset < batch_size * value_dim)

@torch.fx.wrap
def optimized_attention_forward(in_0, in_1):
    """Optimized attention computation that eliminates dropout and fuses operations"""
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]  # This should be 1 based on our tensor shapes
    value_dim = in_1.shape[2]
    
    # Compute output size: [1, 1, batch_size * value_dim]
    output_size = batch_size * value_dim
    out = torch.empty(1, 1, output_size, dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    grid = (batch_size,)
    
    optimized_attention_kernel[grid](
        in_0,
        in_1,
        out,
        batch_size,
        seq_len,
        value_dim,
        BLOCK_SIZE=min(256, value_dim),
    )
    
    return out

def replacement_func():
    return optimized_attention_forward