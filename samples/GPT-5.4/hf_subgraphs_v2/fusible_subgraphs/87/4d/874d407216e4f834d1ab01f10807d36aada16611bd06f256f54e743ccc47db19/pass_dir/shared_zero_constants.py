import torch


GAE_ZERO_FP32 = torch.zeros((1000, 16), dtype=torch.float32, device='cuda')
GAE_ZERO_BF16 = torch.zeros((1000, 16), dtype=torch.bfloat16, device='cuda')
GAE_ZERO_FP16 = torch.zeros((1000, 16), dtype=torch.float16, device='cuda')
RECT_ZERO_FP32 = torch.zeros((128, 128), dtype=torch.float32, device='cuda')
RECT_ZERO_BF16 = torch.zeros((128, 128), dtype=torch.bfloat16, device='cuda')
RECT_ZERO_FP16 = torch.zeros((128, 128), dtype=torch.float16, device='cuda')


@torch.fx.wrap
def zero_dispatch(x, route):
    if route == 0:
        if x.dtype == torch.float32:
            return GAE_ZERO_FP32
        if x.dtype == torch.bfloat16:
            return GAE_ZERO_BF16
        return GAE_ZERO_FP16

    if x.dtype == torch.float32:
        return RECT_ZERO_FP32
    if x.dtype == torch.bfloat16:
        return RECT_ZERO_BF16
    return RECT_ZERO_FP16


def shared_replacement_func():
    return zero_dispatch