"""
Probe pass: test if torch.conv2d (high-level) is in the compiled FX graph.
Uses the exact same call signature as model.py.
If this matches, the graph preserves torch-level ops.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _probe_add_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


@torch.fx.wrap
def _probe_conv_identity(in_10, in_8, in_7):
    """Compute 1x1 conv via Triton GEMM (probe replacement)."""
    device = in_10.device
    dtype = in_10.dtype
    w = in_8.to(device=device, dtype=dtype)
    b = in_7.to(device=device, dtype=dtype)
    N, Cin, H, W = in_10.shape
    Cout = w.shape[0]
    HW = H * W
    A = w.view(Cout, Cin)
    B = in_10.view(Cin, HW)
    out = torch.empty(Cout, HW, dtype=dtype, device=device)
    n = out.numel()
    BLOCK = 1024
    _probe_add_kernel[(n + BLOCK - 1) // BLOCK,](B, out, n, BLOCK=BLOCK)
    return out.view(N, Cout, H, W)


def pattern(in_10, in_8, in_7):
    """Match torch.conv2d with stride=1, pad=0, dil=1, groups=1, non-None bias."""
    return torch.conv2d(in_10, in_8, in_7, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(in_10, in_8, in_7):
    return (in_10, in_8, in_7)


def replacement_func():
    return _probe_conv_identity