import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def bn_relu_elementwise_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused batch_norm (inference) + ReLU kernel.

    Input x is in NCHW layout.
    For each element at flat index i:
        c = (i // (H*W)) % C
        y = ReLU( (x - mean[c]) / sqrt(var[c] + eps) * weight[c] + bias[c])
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    HW = H * W
    # Convert to float32 for numerical stability
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute channel index from NCHW flat index
    c_idx = (offsets // HW) % C

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)

    # batch_norm inference: y = (x - mean) / sqrt(var + eps) * weight + bias
    # Rewritten as: y = x * scale + shift  where
    #   scale = weight / sqrt(var + eps)
    #   shift = bias  - mean * scale
    scale = w * tl.rsqrt(var + 0.001)
    shift = b - mean * scale

    out = x * scale + shift

    # ReLU
    out = tl.where(out > 0.0, out, 0.0)

    # Store in original dtype
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def bn_relu_triton(x, running_mean, running_var, weight, bias):
    """
    Fused batch_norm (inference) + ReLU using a Triton kernel.

    Args:
        x            : input tensor, NCHW layout
        running_mean : shape [C]
        running_var  : shape [C]
        weight       : shape [C]  (gamma)
        bias         : shape [C]  (beta)
    Returns:
        Tensor of same shape/dtype as x.
    """
    N, C, H, W = x.shape
    N_elem = N * C * H * W
    out = torch.empty_like(x)

    def grid(meta):
        return ((N_elem + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    bn_relu_elementwise_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        N, C, H, W,
    )
    return out