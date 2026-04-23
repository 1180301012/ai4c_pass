import torch
import triton
import triton.language as tl
from torch.fx import symbolic_trace
import graph_net_bench.torch.custom_replacement as _custom_replacement


# Make pattern tracing preserve kwargs so it matches the FX graph emitted by the benchmark.
_custom_replacement.force_args_symbolic_trace = symbolic_trace


def pattern(tmp_9, in_3, in_4, in_5):
    tmp_10 = in_3.expand(1, -1, -1)
    tmp_11 = in_4.expand(1, -1, -1)
    tmp_12 = torch.cat((tmp_10, tmp_9, tmp_11), dim=1)
    tmp_13 = in_5[(slice(None, None, None), 0, slice(None, None, None))]
    tmp_14 = tmp_13[(slice(None, None, None), None)]
    tmp_15 = in_5[(slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_16 = in_5[(slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_17 = tmp_16.transpose(1, 2)
    tmp_18 = tmp_17.view(1, 32, 15, 15)
    tmp_19 = torch.nn.functional.interpolate(tmp_18, size=(15, 15), mode='bicubic', align_corners=False)
    tmp_20 = tmp_19.flatten(2)
    tmp_21 = tmp_20.transpose(1, 2)
    tmp_22 = torch.cat((tmp_14, tmp_21, tmp_15), dim=1)
    tmp_23 = tmp_12 + tmp_22
    tmp_24 = torch.nn.functional.dropout(tmp_23, 0.1, False, False)
    return tmp_24


def replacement_args(tmp_9, in_3, in_4, in_5):
    return (tmp_9, in_3, in_4, in_5)


@triton.jit
def _cls_token_add_kernel(cls_ptr, pos_ptr, out_ptr):
    c = tl.arange(0, 32)
    cls_v = tl.load(cls_ptr + c)
    pos_v = tl.load(pos_ptr + c)
    tl.store(out_ptr + c, cls_v + pos_v)


@triton.jit
def _patch_token_add_kernel(tok_ptr, pos_ptr, out_ptr):
    pid = tl.program_id(0)
    c = tl.arange(0, 32)
    tok_v = tl.load(tok_ptr + pid * 32 + c)
    pos_v = tl.load(pos_ptr + (1 + pid) * 32 + c)
    tl.store(out_ptr + (1 + pid) * 32 + c, tok_v + pos_v)


@triton.jit
def _det_token_add_kernel(det_ptr, pos_ptr, out_ptr):
    pid = tl.program_id(0)
    c = tl.arange(0, 32)
    det_v = tl.load(det_ptr + pid * 32 + c)
    pos_v = tl.load(pos_ptr + (226 + pid) * 32 + c)
    tl.store(out_ptr + (226 + pid) * 32 + c, det_v + pos_v)


@torch.fx.wrap
def yolos_token_assembly(tmp_9, in_3, in_4, in_5):
    out = torch.empty((1, 236, 32), device=tmp_9.device, dtype=tmp_9.dtype)

    _cls_token_add_kernel[(1,)](
        cls_ptr=in_3,
        pos_ptr=in_5,
        out_ptr=out,
        num_warps=1,
    )
    _patch_token_add_kernel[(225,)](
        tok_ptr=tmp_9,
        pos_ptr=in_5,
        out_ptr=out,
        num_warps=1,
    )
    _det_token_add_kernel[(10,)](
        det_ptr=in_4,
        pos_ptr=in_5,
        out_ptr=out,
        num_warps=1,
    )
    return out


def replacement_func():
    return yolos_token_assembly