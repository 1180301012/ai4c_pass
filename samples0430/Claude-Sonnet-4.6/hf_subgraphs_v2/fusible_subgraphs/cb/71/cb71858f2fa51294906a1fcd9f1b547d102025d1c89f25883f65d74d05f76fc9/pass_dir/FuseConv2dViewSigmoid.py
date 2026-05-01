"""
Fuse conv2d(stride=1,pad=0,dil=1,groups=1) + view(1,2,8,8) + sigmoid
into a single Triton GEMV+sigmoid kernel.

For in_2=[1,2,1,8], in_1=[128,2,1,8], in_0=[128] the conv2d is exactly
a 128-output GEMV: out[m] = sum_k(w[m,k]*x[k]) + b[m], then sigmoid.
We avoid cuDNN overhead and materialise no intermediate buffer.
"""
import torch
import triton
import triton.language as tl


# ─── pattern ────────────────────────────────────────────────────────────────
def pattern(x, weight, bias):
    conv2d = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    t = conv2d.view(1, 2, 8, 8)
    out = t.sigmoid()
    return out


def replacement_args(x, weight, bias):
    return (x, weight, bias)


# ─── Triton kernel: fused GEMV + sigmoid ─────────────────────────────────────
# W[M,K] @ x[K] + b[M] → sigmoid → out[M]
# M=128, K=16  hardcoded (removes constexpr args from runtime cache-key)
@triton.jit
def _gemv_sigmoid_kernel(
    x_ptr,      # [16] contiguous
    w_ptr,      # [128, 16] contiguous row-major
    b_ptr,      # [128] contiguous
    out_ptr,    # [128] output (addressed as [1,2,8,8])
):
    m = tl.arange(0, 128)   # all 128 outputs
    k = tl.arange(0, 16)    # all 16 inputs

    # Row-major contiguous block load  W [128, 16]
    w = tl.load(w_ptr + m[:, None] * 16 + k[None, :]).to(tl.float32)

    # Load input x [16] and broadcast to [1, 16]
    x    = tl.load(x_ptr + k).to(tl.float32)
    x_2d = tl.expand_dims(x, 0)              # [1, 16]

    # Load bias [128]
    b = tl.load(b_ptr + m).to(tl.float32)

    # GEMV + bias + sigmoid
    acc = tl.sum(w * x_2d, axis=1) + b       # [128]
    tl.store(out_ptr + m, tl.sigmoid(acc))


# ─── wrapper ────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fuse_conv2d_view_sigmoid(x, weight, bias):
    # x      : [1, 2, 1, 8]   (16 elements)
    # weight : [128, 2, 1, 8] (128×16 elements, contiguous)
    # bias   : [128]
    # out    : [1, 2, 8, 8]   (same 128 elements as conv2d output)
    out = torch.empty((1, 2, 8, 8), dtype=x.dtype, device=x.device)
    _gemv_sigmoid_kernel[(1,)](x, weight, bias, out)
    return out


def replacement_func():
    return fuse_conv2d_view_sigmoid