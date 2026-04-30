import torch


def pattern(in_0, in_1, in_2):
    """Pattern: dropout(in_2, p=0.0, training=False) + to(float16) + linear(cast_result, in_1, in_0)
    
    This matches the RECT_L float16 computation where:
    - dropout with p=0.0 and training=False is an identity operation
    - .to(torch.float16) on a float16 tensor is also identity
    - The result feeds into a linear layer
    """
    tmp_2 = torch.nn.functional.dropout(in_2, p = 0.0, training = False)
    to = tmp_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return (linear,)


def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel.
    
    Since dropout(p=0.0, training=False) and .to(same_dtype) are both identity,
    we pass the original input directly to the linear kernel.
    """
    return (in_2, in_1, in_0)


def replacement_func():
    from pass_dir.fused_linear_kernel import fused_linear
    return fused_linear