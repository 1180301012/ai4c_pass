import torch
from pass_dir.shared_kernel import run_fused_conv2d_mean


# ============================================================================
# Pass: Fuse Conv2D + Mean (stride=1, groups=256)
# ============================================================================

def pattern(in_0, in_1):
    """
    Match: conv2d (stride=1, groups=256) followed by mean over spatial dimensions.
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 256)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return conv2d, tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1, "stride1_g256")


@torch.fx.wrap
def replacement_func():
    def dispatch_wrapper(in_0, in_1, route):
        if route == "stride1_g256":
            return run_fused_conv2d_mean(in_0, in_1, 1, 1, route)
        elif route == "stride1_g384":
            return run_fused_conv2d_mean(in_0, in_1, 1, 1, route)
        elif route == "stride1_g768":
            return run_fused_conv2d_mean(in_0, in_1, 1, 1, route)
        elif route == "stride2_g256":
            return run_fused_conv2d_mean(in_0, in_1, 2, 2, route)
        elif route == "stride2_g384":
            return run_fused_conv2d_mean(in_0, in_1, 2, 2, route)
        else:
            # Fallback - should not reach here
            raise ValueError(f"Unknown route: {route}")
    
    return dispatch_wrapper