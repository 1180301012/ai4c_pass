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
    return (tmp_5,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_kernel(
    sigmoid_in_ptr,
    feature_ptr,
    out_ptr,
    SPATIAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(0)

    # Compute scale = 1 + sigmoid(raw) once per channel
    raw_val = tl.load(sigmoid_in_ptr + pid_c).to(tl.float32)
    scale = 1.0 + tl.sigmoid(raw_val)

    # Spatial offsets - no mask needed since SPATIAL is exact multiple of BLOCK
    s_offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    e_offs = pid_c * SPATIAL + s_offs

    # Load, compute, store - fp32 for accuracy
    feat = tl.load(feature_ptr + e_offs).to(tl.float32)
    result = tl.maximum(feat * scale, 0.0)
    tl.store(out_ptr + e_offs, result)


@torch.fx.wrap
def fused_sigmoid_scale_relu(sigmoid_input, feature_input):
    C = 512
    SPATIAL = 64 * 64  # 4096

    out = torch.empty_like(feature_input)

    # 8 spatial blocks × 512 channels = 4096 programs
    BLOCK = 512
    n_spatial_blocks = SPATIAL // BLOCK

    grid = (n_spatial_blocks, C)

    # Launch with default settings (num_warps=4, num_stages=2)
    fused_kernel[grid](
        sigmoid_in_ptr=sigmoid_input,
        feature_ptr=feature_input,
        out_ptr=out,
        SPATIAL=SPATIAL,
        BLOCK=BLOCK,
    )

    return out


def replacement_func():
    return fused_sigmoid_scale_relu