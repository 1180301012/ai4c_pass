import torch
import triton
import triton.language as tl

@triton.jit
def fused_batchnorm_silu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C, H, W,
    eps: tl.constexpr = 1e-05,
    BLOCK_SIZE: tl.constexpr = 256
):
    n_elements = C * H * W
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = offsets
    channel_idx = idx // (H * W)
    channel_idx = tl.where(channel_idx < C, channel_idx, 0)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0, dtype=tl.bfloat16)
    mean = tl.load(mean_ptr + channel_idx, mask=channel_idx < C, other=0.0, dtype=tl.bfloat16)
    var = tl.load(var_ptr + channel_idx, mask=channel_idx < C, other=0.0, dtype=tl.bfloat16)
    weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < C, other=0.0, dtype=tl.bfloat16)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < C, other=0.0, dtype=tl.bfloat16)

    var_sqrt = tl.sqrt(var + eps)
    normalized = (x - mean) * weight / var_sqrt + bias
    exp_term = tl.exp(-normalized)
    silu = normalized * (1.0 / (1.0 + exp_term))

    tl.store(out_ptr + idx, silu, mask=mask)

@torch.fx.wrap
def fused_batchnorm_silu(in_0, in_1, in_2, in_3, in_4):
    reshaped = in_4.reshape(1, 512, 8, 8)
    n_elements = reshaped.numel()
    out_flat = torch.empty_like(reshaped.view(-1))

    num_programs = (n_elements + 255) // 256
    fused_batchnorm_silu_kernel[num_programs](
        reshaped.view(-1),
        mean_ptr=in_0,
        var_ptr=in_1,
        weight_ptr=in_3,
        bias_ptr=in_2,
        out_ptr=out_flat,
        C=512,
        H=8,
        W=8,
        eps=1e-05,
        BLOCK_SIZE=256
    )
    return out_flat.view(1, 512, 8, 8)

def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 8, 8)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

def replacement_func():
    return fused_batchnorm_silu