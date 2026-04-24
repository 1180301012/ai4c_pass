import torch
from pass_dir.shared_dispatch import dispatch


# ---------------------------------------------------------------------------
# Pattern: avg_pool2d(x, 2, 2, 0, True, False, None)
#   kernel_size=2, stride=2, padding=0, ceil_mode=True, count_include_pad=False
# ---------------------------------------------------------------------------
def pattern(x):
    result = torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)
    return result


def replacement_args(x):
    # Append route string so the shared dispatch wrapper knows which branch to run
    return (x, "avgpool")


def replacement_func():
    return dispatch