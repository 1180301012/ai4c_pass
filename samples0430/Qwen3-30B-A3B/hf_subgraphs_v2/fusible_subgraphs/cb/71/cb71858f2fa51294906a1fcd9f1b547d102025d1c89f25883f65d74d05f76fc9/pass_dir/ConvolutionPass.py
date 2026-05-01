import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_2, in_1, in_0):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

# Argument extraction function
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Triton kernel for convolution
@triton.jit
def conv_kernel(
    in_ptr, weight_ptr, bias_ptr,
    out_ptr,
    B: tl.constexpr, C_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    C_out: tl.constexpr
):
    cid = tl.program_id(0)
    # Load bias for current channel
    bias = tl.load(bias_ptr + cid)
    total = 0.0
    # Compute dot product across input dimensions
    for c in range(C_in):
        for h in range(H_in):
            for w in range(W_in):
                idx_in = c * (H_in * W_in) + h * W_in + w
                idx_weight = cid * (C_in * H_in * W_in) + c * (H_in * W_in) + h * W_in + w
                total += tl.load(in_ptr + idx_in) * tl.load(weight_ptr + idx_weight)
    tl.store(out_ptr + cid, total + bias)

# Kernel wrapper
@torch.fx.wrap
def conv2d_triton(in_2, in_1, in_0):
    B, C_in, H_in, W_in = in_2.shape
    C_out, C_in, H_out, W_out = in_1.shape  # weights shape [C_out, C_in, H_out, W_out]
    out = torch.empty(B, C_out, 1, 1, dtype=in_2.dtype, device=in_2.device)
    conv_kernel[(C_out,)](
        in_2, in_1, in_0,
        out,
        B, C_in, H_in, W_in,
        C_out
    )
    return out

# Replacement function
def replacement_func():
    return conv2d_triton