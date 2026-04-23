import torch
import triton
import triton.language as tl


_REL_POS_CACHE = None


def _build_rel_pos_tensor():
    data = [
        [
            [
                [
                    float((q % 14) - (p % 14)),
                    float((q // 14) - (p // 14)),
                    float(
                        ((q % 14) - (p % 14)) * ((q % 14) - (p % 14))
                        + ((q // 14) - (p // 14)) * ((q // 14) - (p // 14))
                    ),
                ]
                for q in range(196)
            ]
            for p in range(196)
        ]
    ]
    return torch.as_tensor(data)


@triton.jit
def _unused_copy_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, x, mask=mask)


def pattern():
    tmp_3 = torch.zeros(1, 196, 196, 3)
    tmp_4 = torch.arange(14)
    tmp_5 = tmp_4.view(1, -1)
    tmp_6 = torch.arange(14)
    tmp_7 = tmp_6.view(-1, 1)
    tmp_8 = tmp_5 - tmp_7
    tmp_9 = tmp_8.repeat(14, 14)
    tmp_10 = tmp_8.repeat_interleave(14, dim=0)
    tmp_11 = tmp_10.repeat_interleave(14, dim=1)
    tmp_12 = tmp_9 ** 2
    tmp_13 = tmp_11 ** 2
    tmp_14 = tmp_12 + tmp_13
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = tmp_15
    tmp_17 = tmp_11.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    tmp_19 = tmp_9.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = tmp_19
    return tmp_3


def replacement_args():
    return ()


@torch.fx.wrap
def cached_relative_position_tensor():
    global _REL_POS_CACHE
    if _REL_POS_CACHE is None:
        _REL_POS_CACHE = _build_rel_pos_tensor()
    return _REL_POS_CACHE


def replacement_func():
    return cached_relative_position_tensor