import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    return tmp_9

def replacement_args(tmp_0):
    return (tmp_0,)

@torch.fx.wrap
def optimized_fused_reshapes(tmp_0):
    # tmp_0 has shape [1, 133, 133]
    # Final output should be [1, 361, 49] where 361 = 19*19, 49 = 7*7
    
    # Original: [1, 133, 133] -> reshape [1, 19, 7, 19, 7] -> transpose [1, 19, 19, 7, 7] -> reshape [1, 361, 49]
    # We can optimize by combining the reshape and transpose operations
    
    # Instead of three separate operations with intermediate tensors, combine them
    # The transpose swaps dimensions 2 and 3, so we can skip the intermediate tensor
    temp = tmp_0.reshape(1, 19, 19, 7, 7)  # directly go to transposed shape
    output = temp.reshape(1, 361, 49)
    
    return output

def replacement_func():
    return optimized_fused_reshapes