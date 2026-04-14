import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Minimal diagnostic pattern: just torch.conv2d (no view/softmax/unsqueeze).
# If this matches, the graph uses torch.conv2d.
# If it doesn't, the graph likely uses ATen ops (aten.convolution.default).
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    out = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return out


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@torch.fx.wrap
def _dummy_conv(in_0, in_1, in_2):
    B, _, H, W = in_2.shape
    return torch.empty(B, 1, H, W, dtype=in_2.dtype, device=in_2.device)


def replacement_func():
    return _dummy_conv