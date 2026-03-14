import torch

def pattern(in_3, in_4):
    """
    Match the sequence: index -> view -> permute -> contiguous -> unsqueeze
    This pattern corresponds to creating a relative position bias from a table lookup.
    
    Original sequence:
    1. in_3[in_4]: [732, 12] indexed by [38809] -> [38809, 12]
    2. view(197, 197, -1): -> [197, 197, 12]
    3. permute(2, 0, 1): -> [12, 197, 197]
    4. contiguous(): -> [12, 197, 197]
    5. unsqueeze(0): -> [1, 12, 197, 197]
    """
    tmp_2 = in_3[in_4]
    tmp_3 = tmp_2.view(197, 197, -1)
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = tmp_5.unsqueeze(0)
    return tmp_6


def replacement_args(in_3, in_4):
    return (in_3, in_4)


def fused_index_view_permute_wrapper(in_3, in_4):
    """
    Optimized version using efficient tensor operations.
    
    Key optimization: Use reshape with permute in a single expression
    to allow better optimization by torch.compile.
    """
    # Perform indexing
    tmp = in_3[in_4]  # [38809, num_heads]
    
    seq_len = 197
    
    # Use reshape and then permute + unsqueeze in one go
    # This reduces the number of intermediate tensors
    tmp = tmp.reshape(seq_len, seq_len, tmp.shape[1])  # [197, 197, num_heads]
    tmp = tmp.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, 197, 197]
    
    return tmp.contiguous()


def replacement_func():
    return fused_index_view_permute_wrapper