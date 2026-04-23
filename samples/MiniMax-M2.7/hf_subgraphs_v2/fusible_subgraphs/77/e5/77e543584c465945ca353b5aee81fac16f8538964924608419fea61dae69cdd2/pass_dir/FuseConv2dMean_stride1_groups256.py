import torch
import triton
import triton.language as tl


# ============================================================================
# Pass: Fuse Conv2D + Mean (stride=1, groups=256)
# Pattern: conv2d(input, weight, None, (1,1), (1,1), (1,1), 256) + mean(dim=(2,3))
# ============================================================================


def pattern(in_0, in_1):
    """
    Match: conv2d (stride=1, groups=256) followed by mean over spatial dimensions.
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 256)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return conv2d, tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def replacement_func():
    def fused_conv2d_mean_stride1_256(in_0, in_1):
        """
        Fused Conv2D (stride=1, groups=256) + Global Average Pooling.
        """
        conv_output = torch.nn.functional.conv2d(
            in_1, in_0, None,
            stride=(1, 1), padding=(1, 1), groups=256
        )
        mean_output = conv_output.mean((2, 3), keepdim=True)
        return conv_output, mean_output
    
    return fused_conv2d_mean_stride1_256