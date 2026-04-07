import torch
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    sigmoid_ptr,           # [embed_dim, seq_len, seq_len] - position bias
    attn_scores_ptr,       # [n_heads, embed_dim, seq_len, seq_len] - attention scores  
    mask_ptr,              # [seq_len, seq_len] - attention mask
    output_ptr,            # [n_heads, embed_dim, seq_len, seq_len] - final output
    n_heads, embed_dim, seq_len, attn_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate thread position for complete tensor processing
    pid = tl.program_id(0)
    total_elements = n_heads * embed_dim * seq_len * attn_dim
    
    if pid * BLOCK_SIZE >= total_elements:
        return
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate indices for 4D tensor
    head_idx = offsets // (embed_dim * seq_len * attn_dim)
    rem = offsets % (embed_dim * seq_len * attn_dim)
    embed_idx = rem // (seq_len * attn_dim)
    rem = rem % (seq_len * attn_dim)
    seq_idx = rem // attn_dim
    attn_idx = rem % attn_dim
    
    # Load position bias: [embed_dim, seq_len, seq_len]
    bias_offset = embed_idx * seq_len * seq_len + seq_idx * seq_len + attn_idx
    position_bias = tl.load(sigmoid_ptr + bias_offset, mask=mask, other=0.0)
    
    # Load attention scores: [n_heads, embed_dim, seq_len, seq_len]  
    score_offset = head_idx * (embed_dim * seq_len * seq_len) +\
                   embed_idx * (seq_len * seq_len) +\
                   seq_idx * seq_len + attn_idx
    attention_score = tl.load(attn_scores_ptr + score_offset, mask=mask, other=0.0)
    
    # Load attention mask: [seq_len, seq_len]
    mask_offset = seq_idx * seq_len + attn_idx
    attention_mask = tl.load(mask_ptr + mask_offset, mask=mask, other=0.0)
    
    # Fuse the operations: result = attention_score + 16 * sigmoid(position_bias) + mask
    fused_result = attention_score + 16.0 * position_bias + attention_mask
    
    # Store final result - note: actual softmax would require more complex reduction
    tl.store(output_ptr + offsets, fused_result, mask=mask)

@torch.fx.wrap  
def optimized_attention(sigmoid_output, attn_scores, attn_mask):
    # Input shapes
    n_heads, embed_dim, seq_len, attn_dim = attn_scores.shape
    
    # Output has same shape as input for the fused computation
    output = torch.zeros_like(attn_scores)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    total_elements = n_heads * embed_dim * seq_len * attn_dim
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_attention_kernel[grid_size](
        sigmoid_ptr=sigmoid_output,
        attn_scores_ptr=attn_scores,
        mask_ptr=attn_mask,
        output_ptr=output,
        n_heads=n_heads,
        embed_dim=embed_dim,
        seq_len=seq_len,
        attn_dim=attn_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return the fused result - the original code will handle softmax and dropout
    return output

def pattern(sigmoid_output, attn_scores, attn_mask):
    # Original pattern: sigmoid -> scaling -> unsqueeze -> add with attention_scores
    #                     -> unsqueeze -> add with mask -> unsqueeze -> add again
    #                     -> view (softmax and dropout handled by original code)
    tmp_10 = 16 * sigmoid_output
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = attn_scores + tmp_11
    tmp_14 = attn_mask.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_12 + tmp_15
    tmp_17 = attn_mask.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    tmp_20 = tmp_19.view(-1, 12, 64, 64)
    # Note: softmax and dropout are handled by the original code after our replacement
    return tmp_20

def replacement_args(sigmoid_output, attn_scores, attn_mask):
    return (sigmoid_output, attn_scores, attn_mask)

def replacement_func():
    return optimized_attention