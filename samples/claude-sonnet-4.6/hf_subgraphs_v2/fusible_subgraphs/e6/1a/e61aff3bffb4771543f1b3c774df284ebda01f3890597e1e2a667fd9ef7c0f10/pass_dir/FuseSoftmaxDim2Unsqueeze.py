import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: FULL model computation (conv2d + view + softmax + unsqueeze)
# This pattern mirrors the ENTIRE forward() from model.py for view(4, 1, 192).
# We cover the remaining view shapes through additional pass files.
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(4, 1, 192)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return (tmp_5,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fused conv1x1 (output_channels=1) + softmax over spatial dim
#
# in_2 shape : [B, C, H, W]   (NCHW)
# in_1 shape : [1, C, 1, 1]   (1x1 conv weight → C values)
# in_0 shape : [1]             (bias)
# out  shape : [B, 1, N, 1]    where N = H*W
#
# Each block handles ONE batch element.
# Step 1 : compute z[n] = dot(weight, x[b,:,h,w]) + bias  for all n=h*W+w
# Step 2 : numerically-stable softmax over z[0..N-1]
# Step 3 : write output (cast back to input dtype) → [B, N] (later unsqueezed)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64,   'BLOCK_C': 64},   num_warps=2),
        triton.Config({'BLOCK_N': 128,  'BLOCK_C': 64},   num_warps=4),
        triton.Config({'BLOCK_N': 256,  'BLOCK_C': 128},  num_warps=4),
        triton.Config({'BLOCK_N': 512,  'BLOCK_C': 128},  num_warps=8),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_C': 256},  num_warps=8),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_C': 256},  num_warps=16),
        triton.Config({'BLOCK_N': 4096, 'BLOCK_C': 512},  num_warps=16),
    ],
    key=['C', 'N'],
)
@triton.jit
def _fused_conv1x1_softmax_kernel(
    x_ptr,    # [B, C, N]  input (NCHW with N = H*W)
    w_ptr,    # [C]         conv weight (flat)
    b_ptr,    # [1]         bias
    out_ptr,  # [B, N]      output (before final unsqueeze)
    B, C, N,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask    = n_offsets < N

    # ---- Step 1: compute z[0..N-1] via linear projection ----
    z = tl.zeros([BLOCK_N], dtype=tl.float32)
    for c in range(0, C):
        # x[batch, c, :] is a contiguous slice of length N
        x_base = x_ptr + batch_idx * C * N + c * N
        x_vals = tl.load(x_base + n_offsets, mask=n_mask, other=0.0)
        w_val  = tl.load(w_ptr + c)
        z     += x_vals.to(tl.float32) * w_val.to(tl.float32)

    bias = tl.load(b_ptr).to(tl.float32)
    z   += bias

    # ---- Step 2: softmax over the N values ----
    z_max = tl.max(tl.where(n_mask, z, float('-inf')), axis=0)
    z_exp = tl.exp(z - z_max)
    z_exp = tl.where(n_mask, z_exp, 0.0)
    z_sum = tl.sum(z_exp, axis=0)
    z_out = z_exp / z_sum          # float32

    # ---- Step 3: store (cast to original dtype) ----
    # Load one element to get the dtype, cast, and store
    x_sample = tl.load(x_ptr)     # scalar – used only for dtype
    out_base  = out_ptr + batch_idx * N
    tl.store(out_base + n_offsets, z_out.to(x_sample.dtype), mask=n_mask)


@torch.fx.wrap
def _fused_conv1x1_softmax_unsqueeze(in_0, in_1, in_2):
    """
    Fused replacement for:
      conv2d(in_2, in_1, in_0, stride=1, pad=0, dil=1, groups=1)   [1x1 conv, out_ch=1]
      .view(B, 1, N)
      .softmax(dim=2)
      .unsqueeze(-1)

    in_0: bias  [1]
    in_1: weight [1, C, 1, 1]
    in_2: input  [B, C, H, W]
    """
    B, C, H, W = in_2.shape
    N = H * W

    # Flatten spatial dims for the kernel
    x_2d  = in_2.view(B, C, N)             # [B, C, N] — contiguous
    w_1d  = in_1.view(C)                   # [C]
    out   = torch.empty((B, N), device=in_2.device, dtype=in_2.dtype)

    _fused_conv1x1_softmax_kernel[(B,)](
        x_2d, w_1d, in_0, out,
        B, C, N,
    )

    # Reshape to [B, 1, N, 1] to match the original model's output
    return out.view(B, 1, N, 1)


def replacement_func():
    return _fused_conv1x1_softmax_unsqueeze