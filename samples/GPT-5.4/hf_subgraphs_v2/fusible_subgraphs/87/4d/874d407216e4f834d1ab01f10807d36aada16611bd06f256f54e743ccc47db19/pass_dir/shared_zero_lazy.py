import torch


_GAE_ZERO_FP32 = None
_GAE_ZERO_BF16 = None
_GAE_ZERO_FP16 = None
_RECT_ZERO_FP32 = None
_RECT_ZERO_BF16 = None
_RECT_ZERO_FP16 = None


@torch.fx.wrap
def zero_dispatch(x, route):
    global _GAE_ZERO_FP32, _GAE_ZERO_BF16, _GAE_ZERO_FP16
    global _RECT_ZERO_FP32, _RECT_ZERO_BF16, _RECT_ZERO_FP16

    if route == 0:
        if x.dtype == torch.float32:
            if _GAE_ZERO_FP32 is None:
                _GAE_ZERO_FP32 = torch.zeros((1000, 16), dtype=torch.float32, device=x.device)
            return _GAE_ZERO_FP32
        if x.dtype == torch.bfloat16:
            if _GAE_ZERO_BF16 is None:
                _GAE_ZERO_BF16 = torch.zeros((1000, 16), dtype=torch.bfloat16, device=x.device)
            return _GAE_ZERO_BF16
        if _GAE_ZERO_FP16 is None:
            _GAE_ZERO_FP16 = torch.zeros((1000, 16), dtype=torch.float16, device=x.device)
        return _GAE_ZERO_FP16

    if x.dtype == torch.float32:
        if _RECT_ZERO_FP32 is None:
            _RECT_ZERO_FP32 = torch.zeros((128, 128), dtype=torch.float32, device=x.device)
        return _RECT_ZERO_FP32
    if x.dtype == torch.bfloat16:
        if _RECT_ZERO_BF16 is None:
            _RECT_ZERO_BF16 = torch.zeros((128, 128), dtype=torch.bfloat16, device=x.device)
        return _RECT_ZERO_BF16
    if _RECT_ZERO_FP16 is None:
        _RECT_ZERO_FP16 = torch.zeros((128, 128), dtype=torch.float16, device=x.device)
    return _RECT_ZERO_FP16


def shared_replacement_func():
    return zero_dispatch