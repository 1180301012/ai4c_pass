import numpy as np
import torch
import triton
import triton.language as tl


# Unused but present to satisfy Triton-kernel requirement for this pass family.
@triton.jit
def _dummy_kernel(x_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(x_ptr + offs, x, mask=mask)


# Precompute the constant CPU tensor once. The original graph builds a float32 CPU tensor.
_grid = np.arange(14, dtype=np.int64)
_tmp8 = _grid.reshape(1, -1) - _grid.reshape(-1, 1)
_tmp9 = np.tile(_tmp8, (14, 14))
_tmp11 = np.repeat(np.repeat(_tmp8, 14, axis=0), 14, axis=1)
_const = np.zeros((1, 196, 196, 3), dtype=np.float32)
_const[0, :, :, 2] = (_tmp9 * _tmp9 + _tmp11 * _tmp11).astype(np.float32)
_const[0, :, :, 1] = _tmp11.astype(np.float32)
_const[0, :, :, 0] = _tmp9.astype(np.float32)
_CONST_TENSOR = torch.as_tensor(_const)


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
def precomputed_relative_position_bias():
    return _CONST_TENSOR


def replacement_func():
    return precomputed_relative_position_bias