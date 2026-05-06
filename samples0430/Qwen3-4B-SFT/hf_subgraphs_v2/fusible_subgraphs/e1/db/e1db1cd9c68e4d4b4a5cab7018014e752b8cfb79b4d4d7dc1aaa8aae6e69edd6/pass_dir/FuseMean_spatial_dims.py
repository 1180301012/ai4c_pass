import torch
import triton
import triton.language as tl


# Match the spatial mean reduction over dims (2, 3) with keepdim=True
def pattern(tmp_0):
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_1


def replacement_args(tmp_0):
    return (tmp_0,)


@triton.jit
def mean_spatial_kernel(
    input_ptr,
    output_mean_ptr,
    C, HW,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) spatial slice of H*W elements
    pid     = tl.program_id(0)
    nc_base = pid * HW

    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW

    # Load in native dtype, upcast to f32 for numerical stability
    x    = tl.load(input_ptr + nc_base + offs, mask=mask, other=0.0)
    xf   = x.to(tl.float32)

    # Accumulate mean (f32) — Triton auto-casts f32 → output pointer type on store
    acc     = tl.where(mask, xf, 0.0)
    mean_val = tl.sum(acc, axis=0) / HW

    tl.store(output_mean_ptr + pid, mean_val)


@torch.fx.wrap
def triton_mean_spatial(tmp_0):
    N, C, H, W = tmp_0.shape
    HW  = H * W
    dev = tmp_0.device

    # Allocate output in input dtype — Triton auto-converts f32 → bf16/fp16 if needed
    out_mean = torch.empty((N, C), dtype=tmp_0.dtype, device=dev)

    BLOCK_HW = 4096  # next power-of-2 >= 3136 (56*56 = 3136)

    mean_spatial_kernel[(N * C,)](
        tmp_0, out_mean,
        C, HW,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )

    return out_mean


def replacement_func():
    return triton_mean_spatial