import torch

def pattern(in_2, tmp_1, tmp_0):
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_3.reshape(-1, 17, 4096)
    return tmp_4

def replacement_args(in_2, tmp_1, tmp_0):
    return (in_2, tmp_1, tmp_0)

def optimized_fused_conv_reshape(x, weight, bias):
    # Step 1: Do 1x1 conv2d (skip the multiplication by 1.0 and reshape)
    conv_out = torch.conv2d(x, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    
    # Step 2: Reshape from [N, 17, 64, 64] to [N, 17, 4096]
    batch_size = conv_out.shape[0]
    reshaped_out = conv_out.reshape(batch_size, 17, -1)
    
    return reshaped_out

def replacement_func():
    return optimized_fused_conv_reshape