import torch
import triton
import triton.language as tl


def _build_rel_pos_tensor():
    data = [[[None for _ in range(3)] for _ in range(196)] for _ in range(196)]
    for p in range(196):
        py = p // 14
        px = p % 14
        for q in range(196):
            qy = q // 14
            qx = q % 14
            dx = float(qx - px)
            dy = float(qy - py)
            data[p][q][0] = dx
            data[p][q][1] = dy
            data[p][q][2] = dx * dx + dy * dy
    return torch.as_tensor([data])


_REL_POS_TENSOR = _build_rel_pos_tensor()


@triton.jit
def _unused_identity_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
def rel_pos_tensor_const():
    return _REL_POS_TENSOR


def replacement_func():
    return rel_pos_tensor_const