import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches exactly the computation pattern in model.py
# Do not include cleanup statements like `tmp_x = None`
def pattern(in_0, in_1, in_2):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    return tmp_3, tmp_3[(Ellipsis, slice(None, 64, None))]

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Triton kernel for fused operation
@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_ptr,
    batch, height, width,
    BLOCK_SIZE: tl.constexpr
):
    
    # Each block processes one (b, h, w) spatial location
    pid = tl.program_id(0)
    b = pid // (height * width)
    h = (pid % (height * width)) // width
    w = pid % width

    # Calculate tensor pointers at (b, h, w)
    in_0_ptr += b * height * width * 64 + h * width * 64 + w * 64
    in_1_ptr += b * height * width * 64 + h * width * 64 + w * 64
    in_2_ptr += b * height * width * 64 + h * width * 64 + w * 64
    out_ptr += b * height * width * 128 + h * width * 128 + w * 128

    # Load in_0 values (64 elements)
    in_0_vals = tl.load(in_0_ptr + tl.arange(0, 64), mask=tl.arange(0, 64) < 64, other=0.0)

    # Load in_2 values (64 elements)
    in_2_vals = tl.load(in_2_ptr + tl.arange(0, 64), mask=tl.arange(0, 64) < 64, other=0.0)

    # Compute einsum: in_2_vals (64) @ in_1 (64x64) -> result (64)
    einsum_vals = tl.zeros(64, dtype=tl.float32)
    for j in range(64):
        in_1_row = tl.load(
            in_1_ptr + j * 64 + tl.arange(0, 64),
            mask=tl.arange(0, 64) < 64,
            other=0.0
        )
        einsum_vals[j] = tl.dot(in_2_vals, in_1_row)

    # Concatenate: in_0_vals (64) + einsum_vals (64) = 128
    combined = tl.concatenate([in_0_vals, einsum_vals])

    # Compute softmax on the combined vector (128 elements)
    max_val = tl.max(combined)
    exp_vals = tl.exp(combined - max_val)
    sum_exp = tl.sum(exp_vals)
    softmax_vals = exp_vals / sum_exp

    # Store result (128 elements)
    tl.store(
        out_ptr + tl.arange(0, 128),
        softmax_vals,
        mask=tl.arange(0, 128) < 128
    )

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_einsum_concat_softmax(in_0, in_1, in_2):
    batch, _, height, width = in_0.shape
    last_dim = 128  # 64 (from in_0) + 64 (from einsum)

    # Allocate output tensor (128 last dimension)
    out = torch.empty(
        batch, height, width, last_dim,
        dtype=in_0.dtype,
        device=in_0.device
    )

    # Calculate number of blocks (one per (b, h, w) location)
    num_blocks = batch * height * width

    # Launch kernel with optimized block size
    fused_kernel[(num_blocks,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        batch=batch,
        height=height,
        width=width,
        BLOCK_SIZE=64  # Optimized block size for 128 elements
    )

    # Extract the first 64 elements for the second return value
    return out, out[(Ellipsis, slice(None, 64, None))]

# Replacement function (NO arguments)
def replacement_func():
    return fused_einsum_concat_softmax