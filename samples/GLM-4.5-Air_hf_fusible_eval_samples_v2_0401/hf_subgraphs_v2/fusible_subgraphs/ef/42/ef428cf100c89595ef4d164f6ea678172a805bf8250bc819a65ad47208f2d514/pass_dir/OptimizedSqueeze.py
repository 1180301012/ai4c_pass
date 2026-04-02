import torch
import triton
import triton.language as tl

def pattern(matmul):
    """
    Pattern matches just the squeeze operation.
    Expects: tmp_1 = matmul.squeeze(1)
    """
    return matmul.squeeze(1)

def replacement_args(matmul):
    return (matmul,)

@triton.jit
def optimized_squeeze_kernel(
    input_ptr,
    output_ptr,
    K: tl.constexpr,  # Size of the last dimension
):
    """Simple kernel to remove dimension 1 from [1, 1, K] → [1, K]"""
    # Program ID  
    pid = tl.program_id(0)
    
    # Since we're removing dimension 1, the output is 1D indexing
    offset = pid * K + tl.arange(0, K)
    mask = offset < K
    
    # Load from [1, 1, K] -> reshape input to [1, K] conceptually
    # In memory, we can treat this as a 1D slice
    input_data = tl.load(input_ptr + offset, mask=mask, other=0.0)
    
    # Store directly to output  
    tl.store(output_ptr + offset, input_data, mask=mask)

@torch.fx.wrap
def optimized_squeeze(matmul):
    # Remove dimension 1
    # Input: [1, 1, 64] → Output: [1, 64]
    
    # For our specific case, squeeze dimension 1
    squeezed = matmul.squeeze(1)
    
    # The squeeze operation is already very efficient in PyTorch
    # This pass shows how you could implement it with Triton if needed,
    # but in practice, the built-in squeeze is already optimized
    
    return squeezed

def replacement_func():
    return optimized_squeeze