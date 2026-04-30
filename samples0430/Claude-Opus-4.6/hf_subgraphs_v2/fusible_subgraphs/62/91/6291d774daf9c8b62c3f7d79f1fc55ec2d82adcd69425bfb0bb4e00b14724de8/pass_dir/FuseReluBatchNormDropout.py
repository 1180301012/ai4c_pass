import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_4, in_0, in_1, in_2, in_3)


@triton.jit
def _fused_relu_bn_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_R
    col_offsets = tl.arange(0, C)

    # Load BN params once per program (shared across rows)
    mean = tl.load(mean_ptr + col_offsets).to(tl.float32)
    var = tl.load(var_ptr + col_offsets).to(tl.float32)
    weight = tl.load(weight_ptr + col_offsets).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets).to(tl.float32)

    # Precompute scale and offset for efficiency
    inv_std = 1.0 / tl.sqrt(var + 1e-05)
    scale = weight * inv_std
    offset = bias - scale * mean

    # Process BLOCK_R rows
    for r in range(BLOCK_R):
        row_idx = row_start + r
        if row_idx < N:
            base = row_idx * C + col_offsets

            # Load input
            x = tl.load(x_ptr + base)
            x_f32 = x.to(tl.float32)

            # ReLU
            x_f32 = tl.maximum(x_f32, 0.0)

            # Fused BatchNorm: scale * x + offset
            out = scale * x_f32 + offset

            # Store with original dtype
            tl.store(out_ptr + base, out.to(x.dtype))


@torch.fx.wrap
def fused_relu_batchnorm(x, running_mean, running_var, bias, weight):
    N = x.shape[0]
    C = x.shape[1]
    device = x.device
    out = torch.empty_like(x)

    # Move params to correct device if needed
    mean_d = running_mean.to(device)
    var_d = running_var.to(device)
    weight_d = weight.to(device)
    bias_d = bias.to(device)

    BLOCK_R = 4
    grid = ((N + BLOCK_R - 1) // BLOCK_R,)

    _fused_relu_bn_kernel[grid](
        x, mean_d, var_d, weight_d, bias_d, out,
        N, C,
        BLOCK_R=BLOCK_R,
        num_warps=4,
    )
    return out


def replacement_func():
    return fused_relu_batchnorm