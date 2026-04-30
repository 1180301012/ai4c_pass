import torch
import triton
import triton.language as tl


@triton.jit
def fused_roll_layernorm_add_kernel(
    input_ptr,    # contiguous input tensor (flat buffer of size N*C)
    in_2_ptr,     # residual [1, N, C] (contiguous)
    weight_ptr,   # weight [C] (contiguous)
    bias_ptr,     # bias [C] (contiguous)
    out_ptr,      # output [1, N, C] (contiguous)
    N,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    SHIFT_H = 4
    SHIFT_W = 4
    EPS = 1e-5

    row_idx = tl.program_id(0)

    if row_idx >= N:
        return

    # Compute rolled source row index
    # torch.roll with shifts=(4,4), dims=(1,2) means:
    # output[r][c] = input[(r - shift_h) % H][(c - shift_w) % W]
    # Use (r + H - SHIFT_H) % H to avoid C-style negative modulo
    r = row_idx // W
    c = row_idx % W
    r_src = (r + H - SHIFT_H) % H
    c_src = (c + W - SHIFT_W) % W
    src_row = r_src * W + c_src  # flat row index in the [H, W] spatial grid

    # First pass: compute mean and sum of squares
    acc_sum = 0.0
    acc_sum_sq = 0.0

    for block_start in tl.range(0, C, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < C

        # Read from contiguous input at rolled source position
        src_offsets = src_row * C + offsets
        x = tl.load(input_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)

        acc_sum += tl.sum(x, axis=0)
        acc_sum_sq += tl.sum(x * x, axis=0)

    mean = acc_sum / C
    # var = E[x^2] - E[x]^2
    variance = acc_sum_sq / C - mean * mean
    rstd = 1.0 / tl.sqrt(variance + EPS)

    # Second pass: normalize, scale, shift, add residual, store
    for block_start in tl.range(0, C, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < C

        # Read from contiguous input at rolled source position
        src_offsets = src_row * C + offsets
        x = tl.load(input_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)

        # Load weight and bias (1D, contiguous)
        w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Load residual (contiguous [1, N, C])
        res_offsets = row_idx * C + offsets
        res = tl.load(in_2_ptr + res_offsets, mask=mask, other=0.0).to(tl.float32)

        # Normalize: (x - mean) * rstd
        x_norm = (x - mean) * rstd

        # Scale and shift: x_norm * weight + bias
        out_ln = x_norm * w + b

        # Add residual
        out = out_ln + res

        # Store result (contiguous [1, N, C])
        tl.store(out_ptr + res_offsets, out, mask=mask)


def _kernel_impl(in_0, in_1, in_2, input_tensor, H_val, W_val, C_val, BLOCK_SIZE_val):
    """Shared kernel implementation for all route variants."""
    N = in_2.shape[1]
    out = torch.empty_like(in_2)
    grid = (N,)
    fused_roll_layernorm_add_kernel[grid](
        input_ptr=input_tensor,
        in_2_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        out_ptr=out,
        N=N,
        H=H_val,
        W=W_val,
        C=C_val,
        BLOCK_SIZE=BLOCK_SIZE_val,
    )
    return out


@torch.fx.wrap
def dispatch_wrapper(in_0, in_1, in_2, input_tensor, route):
    if route == "route_32_32_768":
        return _kernel_impl(in_0, in_1, in_2, input_tensor, H_val=32, W_val=32, C_val=768, BLOCK_SIZE_val=256)
    elif route == "route_64_64_384":
        return _kernel_impl(in_0, in_1, in_2, input_tensor, H_val=64, W_val=64, C_val=384, BLOCK_SIZE_val=256)
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return dispatch_wrapper