"""
Pass: FuseParallelConvMean_B2

Matches the full subgraph for B=2:
  conv2d(in_3, in_1, in_0, ...) -> view(2, 256, -1)   [depends on in_3]
  in_2.mean(dim=-2, keepdim=True)                       [depends on in_2]
  return (mean_result, view_result)

Both outputs are independent. We launch mean on a side CUDA stream so it
overlaps with conv2d on the default stream, hiding the mean latency entirely.
"""

import torch

# Module-level stream – created once, reused across calls.
_stream = None

def _get_stream():
    global _stream
    if _stream is None:
        _stream = torch.cuda.Stream()
    return _stream


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.view(2, 256, -1)
    tmp_4  = in_2.mean(dim=-2, keepdim=True)
    return (tmp_4, tmp_3)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Replacement: parallel conv2d || mean
# ---------------------------------------------------------------------------

@torch.fx.wrap
def parallel_conv_mean_B2(in_0, in_1, in_2, in_3):
    stream = _get_stream()

    # Launch mean on the side stream first (it is independent of conv2d).
    with torch.cuda.stream(stream):
        tmp_4 = in_2.mean(dim=-2, keepdim=True)

    # Launch conv2d on the default stream – overlaps with the mean above.
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.view(2, 256, -1)

    # Wait for the side stream (mean) before returning.
    torch.cuda.current_stream().wait_stream(stream)

    return (tmp_4, tmp_3)


def replacement_func():
    return parallel_conv_mean_B2