import torch

def pattern(in_0, in_1, in_2):
    # First bmm (Q @ K transpose) -> attention scores
    bmm = torch.bmm(in_0, in_1)
    # Softmax normalization
    tmp_1 = torch.nn.functional.softmax(bmm, dim=-1)
    bmm = None
    # Dropout (p=0.0 is identity - can be eliminated)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    tmp_1 = None
    # Second bmm (attention @ V) -> output
    bmm_1 = torch.bmm(tmp_2, in_2)
    tmp_2 = in_2 = None
    # View + Transpose + Reshape can be fused into single Reshape
    tmp_4 = bmm_1.view(1, 8, 1, 32)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 256)
    tmp_5 = None
    return bmm_1, tmp_4, tmp_5, tmp_6

def replacement_args(in_0, in_1, in_2):
    # Extract arguments needed for the fused attention + reshape operation
    return (in_0, in_1, in_2)

@torch.fx.wrap
def fused_attention_reshape_wrapper(in_0, in_1, in_2):
    """
    Fused attention computation with optimized reshape:
    1. Q @ K -> attention scores
    2. softmax(attention scores)
    3. attention @ V -> output
    4. reshape (8, 1, 32) -> (1, 1, 256) directly (fusing view + transpose + reshape)
    """
    # First matmul: Q @ K^T
    bmm = torch.bmm(in_0, in_1)
    
    # Softmax normalization along last dimension
    softmax_scores = torch.nn.functional.softmax(bmm, dim=-1)
    
    # Second matmul: attention @ V (dropout with p=0.0 is identity)
    output = torch.bmm(softmax_scores, in_2)
    
    # Fused reshape: (8, 1, 32) -> (1, 1, 256)
    # Original: view(1, 8, 1, 32) -> transpose(1, 2) -> reshape(1, 1, 256)
    # Equivalent to: reshape(1, 1, 256) since total elements unchanged (256 = 8 * 1 * 32)
    result = output.reshape(1, 1, 256)
    
    return result

def replacement_func():
    return fused_attention_reshape_wrapper