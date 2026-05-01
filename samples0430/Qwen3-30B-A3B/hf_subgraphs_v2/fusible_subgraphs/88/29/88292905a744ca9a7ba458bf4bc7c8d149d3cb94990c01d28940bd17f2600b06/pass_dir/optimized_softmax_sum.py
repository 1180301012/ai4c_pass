import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(in_1, in_0):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(8, -1)
    tmp_2 = tmp_1.view(8, -1, 1, 1)
    tmp_3 = tmp_2.view(8, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    return tmp_5

# Argument extraction function
def replacement_args(in_1, in_0):
    return (in_1, in_0)

# Triton kernel for fused softmax, multiply, and sum
@triton.jit
def fused_softmax_sum_kernel(
    in1_ptr,
    in0_ptr,
    out_ptr,
    B, C, F, H, W,
    BLOCK_H: tl.constexpr = 32,
    BLOCK_W: tl.constexpr = 32,
):
    # Get block IDs
    pid = tl.program_id(0)
    block_h = tl.program_id(1)
    block_w = tl.program_id(2)
    b = pid // F
    f = pid % F

    # Calculate spatial offsets
    h_start = block_h * BLOCK_H
    w_start = block_w * BLOCK_W
    offs_h = tl.arange(0, BLOCK_H)
    offs_w = tl.arange(0, BLOCK_W)
    h = h_start + offs_h
    w = w_start + offs_w

    # Bounds
    valid_h = h < H
    valid_w = w < W
    valid = valid_h & valid_w

    # Load in1 values for this (b,f) along channel dimension
    in1_vals = tl.zeros((C,), dtype=tl.float32)
    for c in range(C):
        in1_vals[c] = tl.load(in1_ptr + b * C * F + c * F + f)

    # Compute softmax
    in1_exp = tl.exp(in1_vals)
    in1_exp_sum = tl.sum(in1_exp)
    softmax_vals = in1_exp / in1_exp_sum

    # Load in0 values for this (b,f) and both channels
    in0_vals = tl.zeros((C,), dtype=tl.float32)
    for c in range(C):
        in0_vals[c] = tl.load(
            in0_ptr + b * C * F * H * W + c * F * H * W + f * H * W + h * W + w,
            mask=valid,
            other=0.0
        )

    # Compute weighted sum over channels
    out_val = 0.0
    for c in range(C):
        out_val += softmax_vals[c] * in0_vals[c]

    # Store result
    tl.store(
        out_ptr + b * F * H * W + f * H * W + h * W + w,
        out_val,
        mask=valid
    )

# Kernel wrapper
@torch.fx.wrap
def fused_softmax_sum(in_1, in_0):
    B = in_1.shape[0]
    C = in_1.shape[1]
    F = in_1.shape[3]  # From [B, C, 1, F] input
    H = in_0.shape[3]
    W = in_0.shape[4]

    out = torch.empty(B, F, H, W, dtype=in_0.dtype, device=in_0.device)

    # Calculate grid dimensions
    BLOCK_H = 32
    BLOCK_W = 32
    grid_h = (H + BLOCK_H - 1) // BLOCK_H
    grid_w = (W + BLOCK_W - 1) // BLOCK_W
    grid = (B * F, grid_h, grid_w)

    # Launch kernel
    fused_softmax_sum_kernel[grid](
        in1_ptr=in_1,
        in0_ptr=in_0,
        out_ptr=out,
        B=B, C=C, F=F, H=H, W=W,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )

    return out

# Replacement function

def replacement_func():
    return fused_softmax_sum