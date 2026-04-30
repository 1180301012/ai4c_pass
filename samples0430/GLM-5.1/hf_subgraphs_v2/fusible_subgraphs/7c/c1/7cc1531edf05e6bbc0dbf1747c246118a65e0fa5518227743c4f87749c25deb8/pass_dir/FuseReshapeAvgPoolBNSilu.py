import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_reshape_avgpool_bn_silu_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    spatial_in,
    spatial_out,
    W_in,
    W_out,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decompose flat index into channel and spatial coordinates
    c = offsets // spatial_out
    spatial = offsets % spatial_out
    h_out = spatial // W_out
    w_out = spatial % W_out

    # Input coordinates for 2x2 average pooling
    h_in = h_out * 2
    w_in_val = w_out * 2

    # Compute flat input indices
    idx_base = c * spatial_in + h_in * W_in + w_in_val

    # Load 4 input values for 2x2 pooling window (compute in float32 for accuracy)
    v0 = tl.load(input_ptr + idx_base, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(input_ptr + idx_base + 1, mask=mask, other=0.0).to(tl.float32)
    v2 = tl.load(input_ptr + idx_base + W_in, mask=mask, other=0.0).to(tl.float32)
    v3 = tl.load(input_ptr + idx_base + W_in + 1, mask=mask, other=0.0).to(tl.float32)

    # Average pooling: (v0 + v1 + v2 + v3) / 4
    pooled = (v0 + v1 + v2 + v3) * 0.25

    # Load batch norm parameters per channel (broadcast across spatial positions)
    mean_val = tl.load(running_mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var_val = tl.load(running_var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    weight_val = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    # Batch norm: output = weight * (input - mean) / sqrt(var + eps) + bias
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    normalized = weight_val * (pooled - mean_val) * inv_std + bias_val

    # SiLU activation: x * sigmoid(x)
    silu_out = normalized * tl.sigmoid(normalized)

    # Store output
    tl.store(output_ptr + offsets, silu_out, mask=mask)


@torch.fx.wrap
def fused_reshape_avgpool_bn_silu(running_mean, running_var, bias, weight, input_tensor):
    C = 512
    H_in = 16
    W_in = 16
    H_out = 8
    W_out = 8
    eps = 1e-05

    spatial_in = H_in * W_in  # 256
    spatial_out = H_out * W_out  # 64
    n_elements = C * spatial_out  # 32768

    output = torch.empty((1, C, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)

    # Grid function that adapts to the autotuned BLOCK_SIZE
    grid = lambda args: (triton.cdiv(args['n_elements'], args['BLOCK_SIZE']),)

    fused_reshape_avgpool_bn_silu_kernel[grid](
        input_ptr=input_tensor,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        spatial_in=spatial_in,
        spatial_out=spatial_out,
        W_in=W_in,
        W_out=W_out,
        eps=eps,
    )

    return output


def replacement_func():
    return fused_reshape_avgpool_bn_silu