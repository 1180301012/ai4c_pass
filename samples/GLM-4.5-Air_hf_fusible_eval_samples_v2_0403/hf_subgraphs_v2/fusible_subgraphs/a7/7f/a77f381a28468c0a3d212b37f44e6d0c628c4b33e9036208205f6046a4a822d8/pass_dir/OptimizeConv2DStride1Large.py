import torch

def pattern(in_1, in_0):
    """
    Match conv2d with stride (1,1) + large slice patterns
    """
    conv2d = torch.conv2d(in_1, in_0, None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    tmp_2 = conv2d[(slice(None, None, None), slice(None, 1024, None), slice(None, None, None), slice(None, None, None))]
    return (tmp_2, conv2d)

def replacement_args(in_1, in_0):
    return (in_1, in_0)

def replacement_func():
    @torch.fx.wrap
    def optimized_conv2d_stride1_large_slice(x, weight, slice_size=1024):
        """
        Optimized convolution with stride (1,1) and large slice operation
        """
        conv2d = torch.conv2d(x, weight, None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
        
        # Apply slice operation with adaptive size
        if slice_size >= 2000:
            actual_slice = 2048
        else:
            actual_slice = slice_size
            
        sliced_output = conv2d[(slice(None, None, None), slice(None, actual_slice, None), slice(None, None, None), slice(None, None, None))]
        
        return sliced_output, conv2d
    
    return optimized_conv2d_stride1_large_slice