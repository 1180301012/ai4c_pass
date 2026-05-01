import torch
import triton
import triton.language as tl

def pattern():
    tmp_10 = torch.arange(11, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(11, dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]
    tmp_14 = tmp_13 - tmp_11
    tmp_15 = -tmp_14
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    tmp_19 += tmp_31
    return tmp_19

def replacement_args():
    return (11,)

@triton.jit
def index_kernel(n, out_ptr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    if i < n and j < n:
        diff = tl.abs(i - j)
        if diff < 8:
            value = diff
        else:
            value = 8.0 + 8.0 * tl.log(diff / 8.0) / 2.772588722239781
            if value > 15.0:
                value = 15.0
        out_val = tl.cast(value, tl.int64)
        tl.store(out_ptr + i * n + j, out_val)

@torch.fx.wrap
def index_kernel_wrapper(n, out_tensor):
    grid = (n, n)
    index_kernel[grid](n, out_tensor.data_ptr())

def replacement_func():
    return index_kernel_wrapper