import torch
import triton
import triton.language as tl

def pattern(matmul, in_2):
    """
    Pattern matching the combined optimization opportunities:
    1. Remove scalar multiplication by 1.0
    2. Remove redundant type conversion from softmax output
    3. Remove dropout no-op
    """
    # Remove scalar multiplication by 1.0 (no-op)
    tmp_1 = matmul * 1.0
    
    # Softmax with type conversion
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    
    # Remove redundant type conversion (softmax already outputs float32)
    tmp_3 = tmp_2.to(torch.float32)
    
    # Remove dropout no-op (p=0.0 is identity function)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    
    return tmp_1, tmp_2, tmp_3, tmp_4

def replacement_args(matmul, in_2):
    return (matmul,)

@torch.fx.wrap
def optimize_attention_sequence(matmul):
    """
    Optimize the entire attention computation sequence:
    1. Skip scalar multiplication by 1.0 (identity function)
    2. Keep softmax (already converts to float32)
    3. Skip redundant type conversion (tmp_2 is already float32)
    4. Skip dropout with p=0.0 (identity function)
    5. Convert back to bfloat16 for next matmul
    """
    # Directly apply softmax - skip scalar mult and type conversions
    softmax_output = torch.nn.functional.softmax(matmul, dim=-1, dtype=torch.float32)
    
    # Convert back to bfloat16 for next computation
    return softmax_output.to(torch.bfloat16)

def replacement_func():
    return optimize_attention_sequence