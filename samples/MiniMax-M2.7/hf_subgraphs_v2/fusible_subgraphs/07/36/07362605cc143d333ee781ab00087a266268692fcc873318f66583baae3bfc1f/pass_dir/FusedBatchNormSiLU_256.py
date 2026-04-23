import torch
import triton
import triton.language as tl

# Pattern matching function - matches reshape -> batch_norm -> silu for 256 channels
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern matches:
    1. Reshape tensor to 4D [1, 256, 16, 16]
    2. Batch normalization with running statistics
    3. SiLU activation (in-place)
    
    This matches graphs with 256 channels and 16x16 spatial size.
    Route: "256_16x16"
    """
    tmp_4 = in_4.reshape(1, 256, 16, 16)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace = True)
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments: running_mean, running_var, bias, weight, input, and route string
    Route "256_16x16" is used to identify this pattern.
    """
    return (in_0, in_1, in_2, in_3, in_4, "256_16x16")

# Import kernels from the main pass file
from pass_dir.FusedBatchNormSiLU import (
    fused_batchnorm_silu_kernel_256,
    fused_batchnorm_silu_kernel_512,
    fused_batchnorm_silu_dispatch
)

def replacement_func():
    """
    Returns the replacement function that implements fused batchnorm + silu.
    Uses routing to select the appropriate kernel.
    """
    return fused_batchnorm_silu_dispatch