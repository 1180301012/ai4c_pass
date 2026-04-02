"""
Pass: FuseParallelConvMean_B16

Matches the full subgraph for B=16 (float16 batch=5):
  conv2d(in_3, in_1, in_0, ...) -> view(16, 256, -1)
  in_2.mean(dim=-2, keepdim=True)
  return (mean_result, view_result)

Both outputs are independent. Mean runs on a side CUDA stream, overlapping
with conv2d on the default stream.
"""

import torch

_stream = None

def _get_stream():
    global _stream
    if _stream is None:
        _stream = torch.cuda.Stream()
    return _stream


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.view(16, 256, -1)
    tmp_4  = in_2.mean(dim=-2, keepdim=True)
    return (tmp_4, tmp_3)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def parallel_conv_mean_B16(in_0, in_1, in_2, in_3):
    stream = _get_stream()

    with torch.cuda.stream(stream):
        tmp_4 = in_2.mean(dim=-2, keepdim=True)

    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.view(16, 256, -1)

    torch.cuda.current_stream().wait_stream(stream)

    return (tmp_4, tmp_3)


def replacement_func():
    return parallel_conv_mean_B16