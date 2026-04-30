import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['N', 'C'],
)
@triton.jit
def _fused_relu_bn_kernel(
    x_ptr,
    rm_ptr,
    rv_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (one sample's feature vector)
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < C

    # Load per-channel BN parameters and promote to float32 for numerical stability
    rm    = tl.load(rm_ptr    + offsets, mask=mask, other=0.0).to(tl.float32)
    rv    = tl.load(rv_ptr    + offsets, mask=mask, other=0.0).to(tl.float32)
    w     = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b     = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

    # Precompute fused BN scale and shift: y = x * scale + shift
    # Where: scale = weight / sqrt(running_var + eps)
    #        shift = bias - running_mean * scale
    scale = w / tl.sqrt(rv + eps)
    shift = b - rm * scale

    # Load row of input, apply ReLU, then BN
    x     = tl.load(x_ptr + row_idx * C + offsets, mask=mask, other=0.0).to(tl.float32)
    relu_x = tl.maximum(x, 0.0)
    out   = relu_x * scale + shift

    # Store result, casting back to input dtype
    tl.store(out_ptr + row_idx * C + offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_relu_bn_dropout(in_0, in_1, in_2, in_3, in_4):
    """
    Fused: ReLU -> BatchNormInference -> Dropout(p=0, training=False)
      in_0 : running_mean  [C]
      in_1 : running_var   [C]
      in_2 : bias  (beta)  [C]
      in_3 : weight(gamma) [C]
      in_4 : input         [N, C]
    """
    N, C = in_4.shape
    out = torch.empty_like(in_4)

    _fused_relu_bn_kernel[(N,)](
        in_4, in_0, in_1, in_3, in_2, out,
        N, C, 1e-5,
    )
    return out


def replacement_func():
    return fused_relu_bn_dropout