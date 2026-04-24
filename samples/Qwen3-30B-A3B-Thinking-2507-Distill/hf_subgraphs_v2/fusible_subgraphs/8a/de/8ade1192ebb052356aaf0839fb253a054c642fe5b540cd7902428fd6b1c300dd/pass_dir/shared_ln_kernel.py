import torch
import triton
import triton.language as tl


# Single-pass kernel: loads x, w, b once; computes mean+var in registers;
# applies scale+shift and stores.  Use power-of-2 BLOCK_SIZE >= hidden_dim.
@triton.jit
def _ln_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    eps,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * hidden_dim

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_dim

    # Load all inputs up-front so the GPU can pipeline memory access
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean (sum over only the valid hidden_dim elements)
    mean = tl.sum(x, axis=0) / hidden_dim

    # Compute variance
    diff = x - mean
    var  = tl.sum(diff * diff, axis=0) / hidden_dim

    # Normalize using hardware rsqrt
    rstd = tl.math.rsqrt(var + eps)

    # Scale + shift, store in original dtype
    out = diff * rstd * w + b
    tl.store(out_ptr + row_start + cols, out.to(x_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def layer_norm_triton(x, weight, bias):
    hidden_dim = x.shape[-1]
    N = x.numel() // hidden_dim
    out = torch.empty_like(x)

    # Fixed configs tuned for common hidden_dim values.
    # BLOCK_SIZE must be a power-of-2 and >= hidden_dim.
    # For hidden_dim=1024 / 768: use 1024 with 8 warps (256 threads, 4 el/thread)
    # For hidden_dim=16:          use   32 with 1 warp  ( 32 threads, 1 el/thread)
    if hidden_dim >= 1024:
        _ln_kernel[(N,)](x, weight, bias, out, N, 1e-5, hidden_dim,
                         BLOCK_SIZE=1024, num_warps=8, num_stages=1)
    elif hidden_dim >= 512:
        _ln_kernel[(N,)](x, weight, bias, out, N, 1e-5, hidden_dim,
                         BLOCK_SIZE=1024, num_warps=8, num_stages=1)
    elif hidden_dim >= 32:
        _ln_kernel[(N,)](x, weight, bias, out, N, 1e-5, hidden_dim,
                         BLOCK_SIZE=64, num_warps=2, num_stages=1)
    else:
        _ln_kernel[(N,)](x, weight, bias, out, N, 1e-5, hidden_dim,
                         BLOCK_SIZE=32, num_warps=1, num_stages=1)

    return out