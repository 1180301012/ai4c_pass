import torch

def pattern(matmul_1, in_2):
    """
    Pattern matching the final operations:
    transposed_matmul -> contiguous -> reshape -> contiguous
    """
    tmp_6 = matmul_1.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9

def replacement_args(matmul_1, in_2):
    return (matmul_1, in_2)



@torch.fx.wrap
def optimized_reshape(matmul_1_input):
    # Input shape is [1, 16, 257, 257] from matmul output
    # The operations are: transpose(1,2) -> contiguous -> reshape(1,257,-1) -> contiguous
    
    # For performance optimization, we can skip no-op operations
    # if the tensor is already in the right shape and layout
    
    # First apply transpose(1,2)
    transposed = matmul_1_input.transpose(1, 2)
    
    # Check if we can skip the contiguous operations
    if transposed.is_contiguous():
        # If already contiguous, just reshape directly
        return transposed.reshape(1, 257, -1)
    else:
        # Otherwise apply reshape and then make contiguous
        reshaped = transposed.reshape(1, 257, -1)
        return reshaped.contiguous()

def replacement_func():
    return optimized_reshape