import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: hardtanh(in_3, 0.0, 6.0) * conv2d_output
# This matches the elementwise post-conv sub-graph present in all target graphs.
# ---------------------------------------------------------------------------
def pattern(conv2d_out, in_3):
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d_out
    return tmp_4


def replacement_args(conv2d_out, in_3):
    return (conv2d_out, in_3)


# ---------------------------------------------------------------------------
# Triton kernel: fused clamp(x, 0, 6) * y  (flat 1-D elementwise)
# Flat because NCHW tensors have the last dim stride-1; the entire tensor is
# one contiguous storage block so consecutive flat offsets work for both inputs.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_hardtanh_mul_kernel(
    x_ptr,        # in_3  – flat [B*C*H*W]
    y_ptr,        # conv2d_out – flat [B*C*H*W]
    out_ptr,      # output  – flat [B*C*H*W]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load y first to overlap fetch latency with x load
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # clamp(x, 0, 6)  ==  hardtanh(0, 6)
    x_clamped = tl.minimum(tl.maximum(x, 0.0), 6.0)

    tl.store(out_ptr + offsets, x_clamped * y, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper — @torch.fx.wrap prevents FX from tracing inside.
# Everything is pre-computed as plain Python ints to minimise per-call overhead.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_hardtanh_mul(conv2d_out, in_3):
    out = torch.empty_like(in_3)

    n_elements = in_3.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    fused_hardtanh_mul_kernel[grid](
        in_3,
        conv2d_out,
        out,
        n_elements,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-argument factory — returns the callable
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_hardtanh_mul