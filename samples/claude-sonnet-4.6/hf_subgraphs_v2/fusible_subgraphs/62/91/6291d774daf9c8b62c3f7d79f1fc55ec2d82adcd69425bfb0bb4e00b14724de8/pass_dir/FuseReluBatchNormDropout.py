import torch
import triton
import triton.language as tl


# Fused kernel: ReLU + BatchNorm (inference) + Dropout (identity, p=0.0 training=False)
# Input: x [N, C], mean [C], var [C], weight [C], bias [C]
# Output: (relu(x) - mean) / sqrt(var + eps) * weight + bias
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=2),
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8),
    ],
    key=['N', 'C'],
)
@triton.jit
def relu_bn_dropout_fused_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N, C,
    BLOCK_C: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_C)
    col_mask = col_offs < C

    # Load batch norm statistics (cached in L2 across rows)
    mean = tl.load(mean_ptr + col_offs, mask=col_mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + col_offs, mask=col_mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr + col_offs, mask=col_mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + col_offs, mask=col_mask, other=0.0).to(tl.float32)

    # Precompute affine coefficients: out = x * scale + shift
    inv_std = tl.rsqrt(var + 1e-5)
    scale   = inv_std * w
    shift   = b - mean * scale

    # Load input row
    x = tl.load(x_ptr + row * C + col_offs, mask=col_mask, other=0.0)

    # ReLU (computed in original dtype)
    x_relu = tl.maximum(x, 0.0)

    # BatchNorm in float32 for numerical stability
    out = x_relu.to(tl.float32) * scale + shift

    # Store result cast back to input dtype
    tl.store(out_ptr + row * C + col_offs, out.to(x.dtype), mask=col_mask)


@torch.fx.wrap
def relu_bn_dropout_fused(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 = running_mean  [C]  (CPU)
    in_1 = running_var   [C]  (CPU)
    in_2 = bias          [C]  (CPU)
    in_3 = weight        [C]  (CPU)
    in_4 = x             [N, C] (CUDA)
    """
    N, C = in_4.shape
    out = torch.empty_like(in_4)

    device = in_4.device

    # Move stats/params to the same device as input
    mean_gpu   = in_0.to(device=device)
    var_gpu    = in_1.to(device=device)
    bias_gpu   = in_2.to(device=device)
    weight_gpu = in_3.to(device=device)

    # One block per row
    relu_bn_dropout_fused_kernel[(N,)](
        in_4, mean_gpu, var_gpu, weight_gpu, bias_gpu, out,
        N, C,
    )

    return out


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return relu_bn_dropout_fused