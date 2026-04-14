import torch

def pattern(in_0, in_1, in_2):
    """Pattern matching conv2d + view + softmax operations"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    # Compute the view shape based on input batch size
    batch_size = in_2.shape[0]
    conv2d = conv2d.view(batch_size, 1, -1)
    
    softmax_out = conv2d.softmax(dim=-1)
    return softmax_out

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement function"""
    return (in_0, in_1, in_2)

def fused_conv2d_view_softmax(in_0, in_1, in_2):
    """
    Fused implementation: conv2d (element-wise) + view + softmax
    
    This combines: torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
                   .reshape(in_2.shape[0], 1, -1)
                   .softmax(dim=-1)
    
    The conv2d with these parameters is equivalent to element-wise operations:
    output = in_2 * in_1 + in_0 (with broadcasting)
    """
    
    # Get tensor shapes and validate
    if in_2.dim() != 4:
        raise ValueError("Input must be 4D tensor [batch, channels, height, width]")
    if in_1.dim() != 4 or in_1.shape[2] != 1 or in_1.shape[3] != 1:
        raise ValueError("Weights must be [1, channels, 1, 1] for 1x1 conv2d")
    if in_0.dim() != 1 or in_0.shape[0] != 1:
        raise ValueError("Bias must be [1] for channel-wise operation")
    
    batch_size = in_2.shape[0]
    channels = in_2.shape[1] 
    height = in_2.shape[2]
    width = in_2.shape[3]
    total_features = channels * height * width
    
    # Step 1: Perform conv2d equivalent (element-wise operations)
    # This is: output = in_2 * in_1 + in_0 with broadcasting
    # For 1x1 conv2d with stride 1, padding 0, equal to element-wise ops
    conv_result = in_2 * in_1 + in_0
    
    # Step 2: Reshape to [batch_size, 1, total_features]
    reshaped = conv_result.reshape(batch_size, 1, total_features)
    
    # Step 3: Apply softmax on last dimension
    result = torch.softmax(reshaped, dim=-1)
    
    return result

@torch.fx.wrap
def replacement_wrapper(in_0, in_1, in_2):
    """Wrapper function for the fused operation"""
    return fused_conv2d_view_softmax(in_0, in_1, in_2)

def replacement_func():
    """Return the fused function"""
    return replacement_wrapper