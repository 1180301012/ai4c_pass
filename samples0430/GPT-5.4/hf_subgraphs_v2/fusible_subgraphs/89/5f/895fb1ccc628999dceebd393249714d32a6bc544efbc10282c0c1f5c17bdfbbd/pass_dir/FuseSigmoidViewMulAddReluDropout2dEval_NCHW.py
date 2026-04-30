import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    tmp_5 = torch.nn.functional.dropout2d(tmp_4, 0.1, False, False)
    return tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _fused_sigmoid_mul_add_relu_kernel(
    gate_ptr,
    x_ptr,
    out_ptr,
    nc,
    BLOCK_HW: tl.constexpr,
    CHANNEL_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    if pid_nc >= nc:
        return

    gate = tl.load(gate_ptr + pid_nc)
    gate = gate.to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate))
    scale = 1.0 + gate

    base = pid_nc * CHANNEL_HW
    for chunk_start in tl.static_range(0, CHANNEL_HW, BLOCK_HW):
        offsets = chunk_start + tl.arange(0, BLOCK_HW)
        x = tl.load(x_ptr + base + offsets)
        x = x.to(tl.float32)
        out = x * scale
        out = tl.maximum(out, 0.0)
        tl.store(out_ptr + base + offsets, out)


@torch.fx.wrap
def fused_sigmoid_view_mul_add_relu_dropout2d_eval_nchw(in_0, in_1):
    out = torch.empty_like(in_1)
    nc = in_0.shape[0] * in_0.shape[1]
    _fused_sigmoid_mul_add_relu_kernel[(nc,)](
        in_0,
        in_1,
        out,
        nc,
        BLOCK_HW=512,
        CHANNEL_HW=4096,
        num_warps=8,
        num_stages=2,
    )
    return out


def replacement_func():
    return fused_sigmoid_view_mul_add_relu_dropout2d_eval_nchw