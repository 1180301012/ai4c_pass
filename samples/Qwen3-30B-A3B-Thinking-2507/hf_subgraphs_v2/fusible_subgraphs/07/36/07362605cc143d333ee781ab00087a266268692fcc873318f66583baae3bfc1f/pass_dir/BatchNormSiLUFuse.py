import torch
import triton
import triton.language as tl

def pattern(tmp_4, in_0, in_1, in_3, in_2):
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6

def replacement_args(tmp_4, in_0, in_1, in_3, in_2):
    return (tmp_4, in_0, in_1, in_3, in_2)

@triton.jit
def batch_norm_silu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    num_elements = C * H * W
    start = tl.program_id(0) * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Calculate channel index and spatial index
    channel_index = offsets // (H * W)
    spatial_index = offsets % (H * W)

    # Load input value
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load channel-specific stats
    mean_val = tl.load(mean_ptr + channel_index, mask=channel_index < C, other=0.0)
    var_val = tl.load(var_ptr + channel_index, mask=channel_index < C, other=0.0)
    weight_val = tl.load(weight_ptr + channel_index, mask=channel_index < C, other=0.0)
    bias_val = tl.load(bias_ptr + channel_index, mask=channel_index < C, other=0.0)

    # Batch normalization
    eps = 1e-05
    denom = tl.sqrt(var_val + eps)
    normed = (x_val - mean_val) / denom
    normed = normed * weight_val + bias_val

    # SiLU activation
    silu_val = normed * tl.sigmoid(normed)

    # Store result
    tl.store(out_ptr + offsets, silu_val, mask=mask)

@torch.fx.wrap
def batch_norm_silu(x, mean, var, weight, bias):
    C = mean.shape[0]
    H = x.shape[2]
    W = x.shape[3]
    num_elements = C * H * W

    out = torch.empty_like(x)

    BLOCK_SIZE = 256
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    batch_norm_silu_kernel[(num_blocks,)](
        x_ptr=x,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

def replacement_func():
    return batch_norm_silu