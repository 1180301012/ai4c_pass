import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern: Conv2d (stride=2) + Reshape + Permute + LayerNorm (C=320)
    
    in_0: layer_norm bias [320]
    in_1: layer_norm weight [320]
    in_2: conv bias [320]
    in_3: conv weight [320, 320, 2, 2]
    in_4: input tensor [B, 320, 32, 32]
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    
    # Conv2d with stride 2
    tmp_4 = torch.conv2d(in_4, tmp_3, tmp_2, (2, 2), (0, 0), (1, 1), 1)
    
    # Reshape to [B, 320, H*W]
    tmp_5 = tmp_4.reshape(32, 320, -1)
    
    # Permute to [B, H*W, 320]
    tmp_6 = tmp_5.permute(0, 2, 1)
    
    # LayerNorm over the last dimension
    tmp_7 = torch.nn.functional.layer_norm(tmp_6, (320,), tmp_1, tmp_0, 1e-05)
    
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments needed for the fused kernel.
    """
    return (in_0, in_1, in_2, in_3, in_4)


@torch.fx.wrap
def fused_kernel_wrapper_320(in_0, in_1, in_2, in_3, in_4):
    """
    Fused implementation: Conv2d + Reshape + Permute + LayerNorm (C=320)
    """
    # Run conv2d
    conv_out = torch.conv2d(in_4, in_3, in_2, (2, 2), (0, 0), (1, 1), 1)
    
    # Reshape
    B = conv_out.shape[0]
    C = conv_out.shape[1]
    conv_flat = conv_out.reshape(B, C, -1)
    
    # Permute
    permuted = conv_flat.permute(0, 2, 1)
    
    # LayerNorm
    output = torch.nn.functional.layer_norm(permuted, (C,), in_1, in_0, 1e-05)
    
    return output


def replacement_func():
    return fused_kernel_wrapper_320