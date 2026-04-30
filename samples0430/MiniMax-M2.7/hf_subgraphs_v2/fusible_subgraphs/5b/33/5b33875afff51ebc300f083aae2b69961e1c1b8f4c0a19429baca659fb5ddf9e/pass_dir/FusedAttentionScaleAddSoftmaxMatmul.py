import torch
import triton
import triton.language as tl
from pass_dir.shared_attention_kernels import shared_attention_dispatcher

# Fused attention kernel: division + add + softmax + dropout + matmul + permute + contiguous
# Pattern matches the attention computation from in_0 through to the output

def pattern(in_0, in_1, in_2, in_3):
    """
    Match the attention computation pattern with scale=8.0.
    """
    # Division (scale by 8.0)
    tmp_0 = in_0 / 8.0
    
    # Add attention mask (broadcasting)
    tmp_1 = tmp_0 + in_2
    
    # Softmax along the last dimension
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    
    # Dropout (p=0.1, training=False, so identity)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    
    # Matmul with value tensor
    matmul = torch.matmul(tmp_3, in_3)
    
    # Permute from (batch, heads, seq, head_dim) to (batch, seq, heads, head_dim)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    
    # Make contiguous
    tmp_6 = tmp_5.contiguous()
    
    # Reshape in_1 (independent path)
    tmp_7 = torch.reshape(in_1, [1, -1, 6, 64])
    
    return (tmp_6, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the fused attention kernel.
    """
    batch, heads, seq_len, _ = in_0.shape
    head_dim = in_3.shape[-1]
    
    # Infer reshape dimensions from tensor shapes
    total_elements = in_1.numel()
    reshape_heads = heads
    reshape_head_dim = total_elements // seq_len // reshape_heads
    
    return (in_0, in_1, in_2, in_3, 8.0, reshape_heads, reshape_head_dim)


def replacement_func():
    """
    Return the shared dispatcher that routes to the appropriate kernel based on scale.
    """
    return shared_attention_dispatcher