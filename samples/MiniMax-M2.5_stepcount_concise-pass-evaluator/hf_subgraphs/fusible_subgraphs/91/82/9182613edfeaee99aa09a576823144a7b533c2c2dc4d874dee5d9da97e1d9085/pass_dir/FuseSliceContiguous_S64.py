import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for the computation with slice optimization.
    
    This pattern matches graphs with S=64:
    - in_0: [1, 1, 64, 64]
    - in_1: [1, 4, 4, 64, 128]
    - in_2: [1, 16, 64, 128]
    - in_3: [1, 16, 64, 128]
    """
    # Slice operation - takes full dimension (64), creates non-contiguous view
    tmp_1 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 64, None)]
    
    # Reshape operation
    tmp_0 = in_1.reshape(1, 16, 64, 128)
    
    # Contiguous calls
    tmp_2 = in_2.contiguous()
    tmp_3 = in_3.contiguous()
    tmp_4 = tmp_0.contiguous()
    
    # Return in same order as original
    return (tmp_1, tmp_3, tmp_2, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    """
    Optimized function that eliminates the redundant slice operation.
    
    The slice in_0[..., :64] is a no-op since in_0 has shape [1, 1, 64, 64].
    By directly using in_0 instead of the sliced view, we avoid creating a 
    non-contiguous tensor that would require a copy when used downstream.
    """
    def optimized_impl(in_0, in_1, in_2, in_3):
        B = in_0.shape[0]
        S = in_0.shape[2]
        
        # Since the slice is a no-op (takes full dimension), directly use in_0
        # This avoids creating a non-contiguous view
        tmp_1 = in_0
        
        # Reshape in_1 to [B, 16, S, 128] 
        tmp_0 = in_1.reshape(B, 16, S, 128)
        
        # Keep the contiguous calls as-is
        tmp_2 = in_2.contiguous()
        tmp_3 = in_3.contiguous()
        tmp_4 = tmp_0.contiguous()
        
        return (tmp_1, tmp_3, tmp_2, tmp_4)
    
    return optimized_impl