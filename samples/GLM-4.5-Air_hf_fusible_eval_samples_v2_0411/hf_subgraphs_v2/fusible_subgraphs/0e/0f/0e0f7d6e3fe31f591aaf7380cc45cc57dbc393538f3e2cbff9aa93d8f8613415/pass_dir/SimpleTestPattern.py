import torch

# Simple test pattern to understand matching
def pattern(in_0, in_1):
    """Simple test pattern"""
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    return (conv2d,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Simple replacement function - just return zeros
def replacement_func():
    def simple_replacement(input, weight):
        # Get input shape
        batch_size, in_channels, in_height, in_width = input.shape
        out_channels, _, kernel_h, kernel_w = weight.shape
        
        # Calculate conv output size
        out_height = (in_height + 2*3 - kernel_h) // 2 + 1
        out_width = (in_width + 2*3 - kernel_w) // 2 + 1
        
        # Return zeros of correct shape
        return torch.zeros((batch_size, out_channels, out_height, out_width), 
                          dtype=input.dtype, device=input.device)
    
    return simple_replacement