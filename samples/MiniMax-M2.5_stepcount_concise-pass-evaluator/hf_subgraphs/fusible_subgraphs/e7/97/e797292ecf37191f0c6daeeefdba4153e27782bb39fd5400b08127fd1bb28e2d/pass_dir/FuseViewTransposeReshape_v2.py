import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Match the pattern: softmax -> dropout -> bmm -> view -> transpose -> reshape
    This is the second graph variant with different dimensions:
    - batch=16, head_dim=64, output=1024
    
    The computation flow for graph 2 (trocr-base-handwritten):
    1. softmax(in_0, dim=-1) -> [16, 1, 1]
    2. dropout(tmp_0, p=0.0, training=False) -> no-op
    3. bmm(tmp_1, in_1) -> [16, 1, 64]
    4. view(1, 16, 1, 64) -> [1, 16, 1, 64]
    5. transpose(1, 2) -> [1, 1, 16, 64]
    6. reshape(1, 1, 1024) -> [1, 1, 1024]
    """
    tmp_0 = torch.nn.functional.softmax(in_0, dim=-1)
    tmp_1 = torch.nn.functional.dropout(tmp_0, p=0.0, training=False)
    tmp_2 = torch.bmm(tmp_1, in_1)
    tmp_3 = tmp_2.view(1, 16, 1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.reshape(1, 1, 1024)
    return tmp_5


def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement kernel."""
    return (in_0, in_1)


@torch.fx.wrap
def fused_attention_wrapper_v2(attn_weights, value_states):
    """
    Fused attention computation that combines:
    - softmax
    - dropout (p=0.0, no-op)
    - batch matmul
    - view + transpose + reshape
    
    This is the same as the first wrapper but with different docstring for v2.
    
    Input shapes:
    - attn_weights: [batch, 1, 1] (e.g., [16, 1, 1])
    - value_states: [batch, 1, head_dim] (e.g., [16, 1, 64])
    
    Output shape: [1, 1, batch * head_dim] (e.g., [1, 1, 1024])
    """
    # Get shapes
    batch_size = attn_weights.shape[0]  # 16
    num_heads = attn_weights.shape[1]   # 1
    head_dim = value_states.shape[2]    # 64
    
    # Compute softmax on attention weights
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # Dropout with p=0.0 is a no-op, skip it
    
    # Batch matrix multiply: [batch, 1, 1] x [batch, 1, head_dim] -> [batch, 1, head_dim]
    # This is equivalent to: attn_probs * value_states (broadcasting)
    context = attn_probs * value_states  # [batch, 1, head_dim]
    
    # Now apply view -> transpose -> reshape in one step
    # Original: view(1, batch, 1, head_dim) -> transpose(1, 2) -> reshape(1, 1, batch*head_dim)
    # Equivalent to: reshape to [1, 1, batch * head_dim] directly
    output = context.reshape(1, 1, batch_size * head_dim)
    
    return output


def replacement_func():
    """Return the replacement function."""
    return fused_attention_wrapper_v2