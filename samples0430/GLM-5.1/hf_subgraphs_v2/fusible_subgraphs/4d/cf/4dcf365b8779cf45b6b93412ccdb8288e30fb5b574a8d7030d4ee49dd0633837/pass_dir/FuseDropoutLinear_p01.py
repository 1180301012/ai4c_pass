import torch


def pattern(in_0, in_1, in_2):
    """Pattern: dropout(in_2, 0.1, False, False) + linear(dropout_result, in_1, in_0)
    
    This matches the bigbird-roberta-base computation where:
    - dropout with p=0.1 and training=False is an identity operation
    - The result feeds directly into a linear layer
    """
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return (linear,)


def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel.
    
    Since dropout with training=False is identity, we pass the original input
    directly to the linear kernel, skipping the intermediate dropout.
    """
    return (in_2, in_1, in_0)


def replacement_func():
    from pass_dir.fused_linear_kernel import fused_linear
    return fused_linear