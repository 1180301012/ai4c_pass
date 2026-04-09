import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # This pattern matches the core computation chain:
    # conv2d -> sigmoid (already optimized by framework) -> view -> multiply -> contiguous
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@torch.fx.wrap
def optimized_computation_chain(in_0, in_1, in_2, in_3):
    """
    Optimization: Remove intermediate tensor assignments and redundant operations.
    Many temporary tensors in PyTorch graphs are unnecessary overhead.
    """
    
    # Perform the core computation in a more efficient chain:
    # 1. Convolution (expensive, must keep)
    conv_out = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    
    # 2. Sigmoid (may be optimized by modern PyTorch automatically)
    sigmoid_out = torch.sigmoid(conv_out)
    
    # 3. Remove unnecessary view operation if possible - handle broadcasting directly
    # The view(1, -1, 1, 1) is often redundant for broadcasting
    # Check if we can avoid creating this intermediate tensor
    if sigmoid_out.ndim == 1:
        # Handle 1D case directly for broadcasting
        expanded_sigmoid = sigmoid_out.reshape(1, -1, 1, 1)
    else:
        expanded_sigmoid = sigmoid_out
    
    # 4. Element-wise multiplication
    mul_out = in_2 * expanded_sigmoid
    
    # 5. Check if contiguous call is needed
    if mul_out.is_contiguous():
        result = mul_out
    else:
        result = mul_out.contiguous()
    
    return result

def replacement_func():
    return optimized_computation_chain