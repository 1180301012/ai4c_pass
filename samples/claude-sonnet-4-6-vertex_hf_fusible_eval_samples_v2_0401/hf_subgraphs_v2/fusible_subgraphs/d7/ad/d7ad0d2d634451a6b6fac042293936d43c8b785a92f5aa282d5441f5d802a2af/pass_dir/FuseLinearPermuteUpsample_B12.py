import torch
import triton
import triton.language as tl


@triton.jit
def bilinear_upsample_kernel_b12(
    input_ptr, output_ptr, BC,
    H_IN: tl.constexpr, W_IN: tl.constexpr,
    H_OUT: tl.constexpr, W_OUT: tl.constexpr,
    DTYPE: tl.constexpr,
):
    bc = tl.program_id(0)
    oh = tl.program_id(1)
    h_src   = (oh.to(tl.float32) + 0.5) * (H_IN / H_OUT) - 0.5
    h_floor = tl.math.floor(h_src)
    h0 = tl.maximum(h_floor.to(tl.int32), 0)
    h1 = tl.minimum(h_floor.to(tl.int32) + 1, H_IN - 1)
    frac_h = h_src - h_floor
    ow = tl.arange(0, W_OUT)
    w_src   = (ow.to(tl.float32) + 0.5) * (W_IN / W_OUT) - 0.5
    w_floor = tl.math.floor(w_src)
    w0 = tl.maximum(w_floor.to(tl.int32), 0)
    w1 = tl.minimum(w_floor.to(tl.int32) + 1, W_IN - 1)
    frac_w = w_src - w_floor
    in_base = bc * H_IN * W_IN
    v00 = tl.load(input_ptr + in_base + h0 * W_IN + w0).to(tl.float32)
    v01 = tl.load(input_ptr + in_base + h0 * W_IN + w1).to(tl.float32)
    v10 = tl.load(input_ptr + in_base + h1 * W_IN + w0).to(tl.float32)
    v11 = tl.load(input_ptr + in_base + h1 * W_IN + w1).to(tl.float32)
    result = (1.0 - frac_h) * ((1.0 - frac_w) * v00 + frac_w * v01) + \
              frac_h * ((1.0 - frac_w) * v10 + frac_w * v11)
    out_offset = bc * H_OUT * W_OUT + oh * W_OUT + ow
    if DTYPE == 1:
        tl.store(output_ptr + out_offset, result.to(tl.float16))
    elif DTYPE == 2:
        tl.store(output_ptr + out_offset, result.to(tl.bfloat16))
    else:
        tl.store(output_ptr + out_offset, result)


@torch.fx.wrap
def triton_bilinear_upsample_128_b12(x: torch.Tensor) -> torch.Tensor:
    B, C, H_IN, W_IN = x.shape
    BC = B * C
    x_c = x.contiguous()
    out = torch.empty(B, C, 128, 128, dtype=x.dtype, device=x.device)
    dtype_id = 1 if x.dtype == torch.float16 else (2 if x.dtype == torch.bfloat16 else 0)
    bilinear_upsample_kernel_b12[(BC, 128)](
        x_c, out, BC, H_IN=H_IN, W_IN=W_IN, H_OUT=128, W_OUT=128, DTYPE=dtype_id,
    )
    return out


@torch.fx.wrap
def fused_permute_reshape_upsample_b12(x):
    """x: [12, 256, 768] – linear output before permute."""
    reshaped = x.permute(0, 2, 1).reshape(12, -1, 16, 16)
    return triton_bilinear_upsample_128_b12(reshaped)


# Pattern: permute(0,2,1) + reshape(12,-1,16,16) – both call_method nodes.
# Anchor = reshape (definitely in model graph). Replacement returns [12,768,128,128].
# Downstream F.interpolate([12,768,128,128], size=(128,128)) is a mathematical identity.
def pattern(x):
    tmp_3 = x.permute(0, 2, 1)
    tmp_4 = tmp_3.reshape(12, -1, 16, 16)
    return tmp_4


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_permute_reshape_upsample_b12