import torch

# Pattern to match: sigmoid -> view -> mul -> contiguous
# This optimizes by removing unnecessary view and contiguous operations
def pattern(conv_output, in_2):
    # Sigmoid on conv2d output
    tmp_3 = torch.sigmoid(conv_output)
    
    # View reshape (no actual change for 4D tensor with same shape)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    
    # Multiply with in_2 (broadcasts across spatial dims)
    tmp_5 = in_2 * tmp_4
    
    # Make contiguous - this is the final observable output
    tmp_6 = tmp_5.contiguous()
    
    # Only return the final output that matches the model's return
    return tmp_6


def replacement_args(conv_output, in_2):
    # Return inputs needed for the optimized implementation
    return (conv_output, in_2)


def optimized_impl(conv_output, in_2):
    """
    Optimized implementation that removes unnecessary operations.
    - The view operation is a no-op for 4D tensors (shape stays the same)
    - The contiguous() call is unnecessary after element-wise ops
    """
    # Use tensor method which is allowed
    sigmoid_val = conv_output.sigmoid()
    
    # Direct multiply - PyTorch broadcasts automatically
    # The result is already contiguous
    result = in_2 * sigmoid_val
    
    return result


def replacement_func():
    return optimized_impl