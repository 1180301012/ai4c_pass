import torch
import triton
import triton.language as tl


def pattern(tmp_2):
    """
    Match permute + contiguous + view and replace with efficient reshape.
    This eliminates the need for contiguous() by using reshape instead.
    """
    tmp_3 = tmp_2.permute(0, 2, 1, 3)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.view(4, 512, 32)
    return tmp_5


def replacement_args(x):
    """
    Extract arguments needed for replacement.
    """
    return (x,)


def replacement_func():
    """
    Returns the replacement function.
    Uses reshape instead of permute + contiguous + view.
    reshape can handle non-contiguous tensors without the explicit contiguous() call.
    """
    def optimized_op(x):
        # Use reshape which handles non-contiguous tensors automatically
        # This is equivalent to: x.permute(0, 2, 1, 3).contiguous().view(4, 512, 32)
        # But more efficient as it avoids the explicit contiguous() call
        return x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], x.shape[1] * x.shape[3])
    
    return optimized_op