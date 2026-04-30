import torch


# Pattern to match: conv2d + inplace_relu
def pattern(in_0, in_1, in_3):
    """Match conv2d followed by inplace ReLU"""
    conv2d = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
    tmp_3 = torch.nn.functional.relu(conv2d, inplace=True)
    return tmp_3


def replacement_args(in_0, in_1, in_3):
    """Extract arguments: bias, weight, input"""
    return (in_0, in_1, in_3)


def replacement_func():
    def fused_conv_relu(in_0, in_1, in_3):
        """Fused Conv2d + ReLU using PyTorch ops for now"""
        conv_out = torch.conv2d(in_3, in_1, in_0, (2, 2), (1, 1), (1, 1), 1)
        return torch.nn.functional.relu(conv_out, inplace=True)
    return fused_conv_relu