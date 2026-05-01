import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Multi-channel BN inference kernel
#
# Grid: (N * ceil_div(C, CPB),)
#   Each program processes CPB consecutive channels for one batch item.
#   The unrolled loop (CPB iterations) amortises block-launch cost:
#   for N=1, C=2432, CPB=8 → 304 blocks instead of 2432.
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def bn_multi_channel_kernel(
    input_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C, HW, eps,
    HW_BLOCK: tl.constexpr,  # next_power_of_2(HW)
    CPB: tl.constexpr,        # channels per block
):
    block_id = tl.program_id(0)

    # Which batch item and which channel group
    c_groups = (C + CPB - 1) // CPB      # runtime scalar ceil-div
    n      = block_id // c_groups
    c_grp  = block_id  % c_groups
    c_base = c_grp * CPB

    offsets  = tl.arange(0, HW_BLOCK)
    sp_mask  = offsets < HW              # spatial validity mask [HW_BLOCK]

    # Unrolled: CPB channel passes, each with its own BN transform
    for local_c in tl.static_range(CPB):
        c       = c_base + local_c          # runtime scalar channel index
        c_valid = c < C                     # handle last partial group

        m = tl.load(mean_ptr   + c, mask=c_valid, other=0.0).to(tl.float32)
        v = tl.load(var_ptr    + c, mask=c_valid, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + c, mask=c_valid, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr   + c, mask=c_valid, other=0.0).to(tl.float32)

        inv_std = 1.0 / tl.sqrt(v + eps)
        scale   = w * inv_std
        shift   = b - m * scale

        # NCHW: element (n, c, ·) starts at (n*C + c) * HW
        base  = ((n * C + c).to(tl.int64)) * HW
        vmask = sp_mask & c_valid          # broadcast scalar → [HW_BLOCK]

        x   = tl.load(input_ptr + base + offsets, mask=vmask, other=0.0).to(tl.float32)
        out = x * scale + shift
        tl.store(output_ptr + base + offsets, out, mask=vmask)


@torch.fx.wrap
def triton_batch_norm_inference(input, running_mean, running_var, weight, bias):
    device = input.device
    dtype  = input.dtype

    m = torch.as_tensor(running_mean, device=device, dtype=dtype)
    v = torch.as_tensor(running_var,  device=device, dtype=dtype)
    w = torch.as_tensor(weight,       device=device, dtype=dtype)
    b = torch.as_tensor(bias,         device=device, dtype=dtype)

    N, C, H, W = input.shape
    HW     = H * W
    output = torch.empty_like(input)

    # HW_BLOCK: SIMD width per channel pass = next power-of-2 ≥ HW (capped 4096)
    HW_BLOCK = max(32, min(4096, triton.next_power_of_2(HW)))
    CPB      = 8   # 8 channels per block → 8× fewer blocks than single-channel

    num_warps = 4 if HW_BLOCK <= 1024 else 8
    grid      = (N * triton.cdiv(C, CPB),)

    bn_multi_channel_kernel[grid](
        input, m, v, w, b, output,
        N, C, HW, 0.001,
        HW_BLOCK=HW_BLOCK,
        CPB=CPB,
        num_warps=num_warps,
    )

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Pattern / replacement interface
# ──────────────────────────────────────────────────────────────────────────────

def pattern(input, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(
        input, running_mean, running_var, weight, bias,
        False, 0.1, 0.001
    )


def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)


def replacement_func():
    return triton_batch_norm_inference