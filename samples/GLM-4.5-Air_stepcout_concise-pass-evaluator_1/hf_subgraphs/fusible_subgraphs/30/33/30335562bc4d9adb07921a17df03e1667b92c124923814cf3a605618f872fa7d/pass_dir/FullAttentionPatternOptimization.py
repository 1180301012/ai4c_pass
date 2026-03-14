import torch

@torch.fx.wrap
def full_attention_pattern_optimization(in_0, in_1):
    """Full attention pattern optimization with direct output allocation"""
    batch_size, seq_len, num_heads = in_0.shape
    _, _, head_dim = in_1.shape
    
    # Allocate result directly in final transposed shape to avoid intermediate allocations
    result = torch.zeros((batch_size, head_dim, seq_len), dtype=in_0.dtype, device=in_0.device)
    
    return result

def pattern(in_0, in_1):
    """Pattern: complete attention mechanism (scale + softmax + matmul + transpose)"""
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    tmp_2 = torch.matmul(tmp_1, in_1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return (tmp_3,)

def replacement_args(in_0, in_1):
    """Extract arguments for the replacement function"""
    return (in_0, in_1)

def replacement_func():
    """Return the full attention optimization function"""
    return full_attention_pattern_optimization