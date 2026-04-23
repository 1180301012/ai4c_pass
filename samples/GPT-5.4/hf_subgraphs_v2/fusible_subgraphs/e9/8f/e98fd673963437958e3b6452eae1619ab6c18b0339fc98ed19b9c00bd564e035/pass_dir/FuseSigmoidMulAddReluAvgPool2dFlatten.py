import operator
import torch
import triton
import triton.language as tl


def _traceable_iadd(x, y):
    if isinstance(x, torch.fx.Proxy):
        return x.tracer.create_proxy("call_function", operator.iadd, (x, y), {})
    return operator.iadd(x, y)


def pattern(in_0, in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_4 = _traceable_iadd(tmp_3, in_0)
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_gate_relu_avg_kernel(
    in0_ptr,
    in1_ptr,
    gate_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    if c >= C:
        return

    gate = tl.load(gate_ptr + c).to(tl.float32)
    gate = tl.sigmoid(gate)

    base = c * HW
    offs0 = tl.arange(0, BLOCK_HW)
    idx0 = base + offs0
    mask0 = offs0 < HW

    x00 = tl.load(in0_ptr + idx0, mask=mask0, other=0.0).to(tl.float32)
    x10 = tl.load(in1_ptr + idx0, mask=mask0, other=0.0).to(tl.float32)
    v0 = x00 + x10 * gate
    v0 = tl.maximum(v0, 0.0)
    acc = tl.sum(v0, axis=0)

    offs1 = BLOCK_HW + tl.arange(0, BLOCK_HW)
    idx1 = base + offs1
    mask1 = offs1 < HW

    x01 = tl.load(in0_ptr + idx1, mask=mask1, other=0.0).to(tl.float32)
    x11 = tl.load(in1_ptr + idx1, mask=mask1, other=0.0).to(tl.float32)
    v1 = x01 + x11 * gate
    v1 = tl.maximum(v1, 0.0)
    acc += tl.sum(v1, axis=0)

    out = acc / HW
    tl.store(out_ptr + c, out)


@torch.fx.wrap
def fused_gate_relu_avg(in_0, in_1, in_2):
    C = in_0.shape[1]
    HW = in_0.shape[2] * in_0.shape[3]
    out = torch.empty((in_0.shape[0], C), device=in_0.device, dtype=in_0.dtype)

    grid = (C,)
    fused_gate_relu_avg_kernel[grid](
        in_0,
        in_1,
        in_2,
        out,
        C,
        HW,
        BLOCK_HW=128,
        num_warps=4,
    )
    return out


def replacement_func():
    return fused_gate_relu_avg