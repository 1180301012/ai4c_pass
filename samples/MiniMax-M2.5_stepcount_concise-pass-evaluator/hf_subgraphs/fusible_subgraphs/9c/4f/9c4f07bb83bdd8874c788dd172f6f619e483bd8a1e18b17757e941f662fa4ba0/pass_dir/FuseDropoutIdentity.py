import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match the pattern of two consecutive dropout operations with p=0.0.
    When p=0.0, dropout is an identity operation, so two consecutive dropouts
    can be replaced with a single identity.
    
    Pattern matches:
    - Extract first element from input tuple
    - First dropout with p=0.0, training=False, inplace=False
    - Second dropout with p=0.0, training=False, inplace=False
    - Return the second dropout output
    """
    tmp_0 = in_0[0]
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return tmp_2


def replacement_args(in_0):
    """
    Extract arguments needed for the replacement function.
    We need the original input tensor (first element of the tuple).
    """
    return (in_0,)


def replacement_func():
    """
    Replacement function that fuses two identity dropouts into a simple passthrough.
    Since dropout with p=0.0 returns the input unchanged, we can just return
    the first input element directly.
    """
    
    @torch.fx.wrap
    def fused_dropout(in_0):
        """
        Optimized replacement: simply return the first element of the input tuple.
        Since dropout with p=0.0 is mathematically identity, two consecutive
        identity operations = single identity = just return the input.
        """
        return in_0[0]
    
    return fused_dropout