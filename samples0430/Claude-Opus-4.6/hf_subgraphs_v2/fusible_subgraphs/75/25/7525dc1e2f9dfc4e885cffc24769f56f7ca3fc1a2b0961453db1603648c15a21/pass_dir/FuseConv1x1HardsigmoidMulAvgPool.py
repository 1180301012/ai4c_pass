import torch
import triton
import triton.language as tl


def pattern(conv_out, x):
    tmp_3 = torch.nn.functional.hardsigmoid(conv_out, False)
    tmp_4 = x * tmp_3
    tmp_5 = torch.nn.functional.adaptive_avg_pool2d(tmp_4, 1)
    tmp_6 = tmp_5.flatten(1, -1)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    return tmp_7


def replacement_args(conv_out, x):
    return (conv_out, x)


@triton.jit
def fused_hardsigmoid_mean_mul_kernel(
    conv_out_ptr,
    x_ptr,
    out_ptr,
    BC,
    HW,
    BLOCK_BC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    bc_start = pid * BLOCK_BC
    bc_offs = bc_start + tl.arange(0, BLOCK_BC)
    bc_mask = bc_offs < BC

    # Load conv_out and apply hardsigmoid: clamp((x + 3) / 6, 0, 1)
    conv_vals = tl.load(conv_out_ptr + bc_offs, mask=bc_mask, other=0.0).to(tl.float32)
    hs_vals = tl.minimum(tl.maximum((conv_vals + 3.0) / 6.0, 0.0), 1.0)

    # 2D load: [BLOCK_BC, BLOCK_HW] and compute row-wise mean
    hw_offs = tl.arange(0, BLOCK_HW)
    x_ptrs = x_ptr + bc_offs[:, None] * HW + hw_offs[None, :]
    x_mask = (bc_offs[:, None] < BC) & (hw_offs[None, :] < HW)
    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
    mean_vals = tl.sum(x_vals, axis=1) / HW

    # Output = hardsigmoid * mean
    results = hs_vals * mean_vals
    tl.store(out_ptr + bc_offs, results, mask=bc_mask)


@torch.fx.wrap
def fused_hardsigmoid_avgpool(conv_out, x):
    B = conv_out.shape[0]
    C = conv_out.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    HW = H * W
    BC = B * C
    dtype = x.dtype
    device = x.device

    out = torch.empty((B, C), dtype=dtype, device=device)
    BLOCK_HW = triton.next_power_of_2(HW)

    # Choose BLOCK_BC to balance parallelism and work-per-program
    if BC <= 2048:
        BLOCK_BC = 4   # B=1: grid=256
    elif BLOCK_HW <= 64:
        BLOCK_BC = 64  # Large BC, small HW: more work per program
    else:
        BLOCK_BC = 8   # Large BC, large HW: less register pressure, higher occupancy

    grid = (triton.cdiv(BC, BLOCK_BC),)
    fused_hardsigmoid_mean_mul_kernel[grid](
        conv_out, x, out,
        BC, HW,
        BLOCK_BC=BLOCK_BC,
        BLOCK_HW=BLOCK_HW,
    )

    return out


def replacement_func():
    return fused_hardsigmoid_avgpool