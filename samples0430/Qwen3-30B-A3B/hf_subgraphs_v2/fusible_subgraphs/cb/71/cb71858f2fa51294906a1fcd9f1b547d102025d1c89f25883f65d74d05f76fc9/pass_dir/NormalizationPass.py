import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_3):
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return tmp_6

# Argument extraction function

def replacement_args(in_3):
    return (in_3,)

# Triton kernel for normalization
@triton.jit
def norm_kernel(in_ptr, out_ptr, B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr):
    # Each block processes one (b, c, h) group
    pid = tl.program_id(0)
    b = pid // (C * H)
    c = (pid // H) % C
    h = pid % H
    # Calculate start index in flattened tensor
    start_idx = b * (C * H * W) + c * (H * W) + h * W
    # Compute sum across W dimension for this group
    sum_val = 0.0
    for i in range(W):
        sum_val += tl.load(in_ptr + start_idx + i)
    # Divide each element by sum_val
    for i in range(W):
        val = tl.load(in_ptr + start_idx + i)
        tl.store(out_ptr + start_idx + i, val / sum_val)

# Kernel wrapper
@torch.fx.wrap
def normalize(in_3):
    B, C, H, W = in_3.shape
    grid_size = B * C * H
    out = torch.empty_like(in_3)
    norm_kernel[grid_size, W](in_3, out, B, C, H, W)
    return out

# Replacement function

def replacement_func():
    return normalize