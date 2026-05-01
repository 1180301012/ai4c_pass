import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Shape-specialized kernel: C=512, HW=4096, one CTA per channel.
# No mask, no autotuning — minimum dispatch overhead.
# out[c, hw] = relu( in_1[c, hw] * (1 + sigmoid(in_0[c])) )
# ---------------------------------------------------------------------------

@triton.jit
def _fused_channel_relu_kernel(
    in0_ptr,              # [C]    – per-channel scale
    in1_ptr,              # [C*HW] – feature map (NCHW, N=1, contiguous)
    out_ptr,              # [C*HW] – output
    BLOCK: tl.constexpr,  # = HW = 4096
):
    c = tl.program_id(0)   # channel index (0..511)

    # per-channel sigmoid in fp32 (1 scalar per CTA)
    s      = tl.load(in0_ptr + c).to(tl.float32)
    factor = 1.0 + tl.sigmoid(s)

    # load the entire channel tile, compute relu(x*(1+sigmoid(s))), store
    base = c * BLOCK
    offs = base + tl.arange(0, BLOCK)   # BLOCK=4096 → compile-time range

    x = tl.load(in1_ptr + offs)
    y = x.to(tl.float32) * factor
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + offs, y.to(x.dtype))


# Pre-create the fixed grid tuple to avoid a Python object allocation per call
_GRID_512 = (512,)


@torch.fx.wrap
def fused_sigmoid_scale_relu(in_0, in_1):
    """
    Fused: out = relu( in_1 * (1 + sigmoid(in_0)) )
    Specialised for in_0=[1,512], in_1=[1,512,64,64].
    512 CTAs (one per channel), BLOCK=4096, num_warps=4 → single execution wave.
    """
    out = torch.empty_like(in_1)
    _fused_channel_relu_kernel[_GRID_512](
        in_0, in_1, out,
        BLOCK=4096,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern, replacement_args, replacement_func required by the framework
# ---------------------------------------------------------------------------

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


def replacement_func():
    return fused_sigmoid_scale_relu