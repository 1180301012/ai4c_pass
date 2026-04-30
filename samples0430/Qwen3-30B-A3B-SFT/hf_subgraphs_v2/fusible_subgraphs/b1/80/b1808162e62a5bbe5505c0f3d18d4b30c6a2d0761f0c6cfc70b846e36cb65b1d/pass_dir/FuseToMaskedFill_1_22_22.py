import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_masked_fill_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Use masked load/store for safety with arbitrary n_elements
    mask = offsets < n_elements

    # Load int64; other=1 gives x_f=1.0 (tmp_1=0.0, harmless) for OOB lanes
    x = tl.load(in_ptr + offsets, mask=mask, other=1)
    x_f = x.to(tl.float32)

    # tmp_1 = 1.0 - x_f
    tmp_1 = 1.0 - x_f

    # masked_fill: value=-3.4e38 where tmp_1 != 0, else 0.0
    tmp_3 = tl.where(tmp_1 != 0.0, -3.4028234663852886e+38, 0.0)

    # tmp_4 = tmp_3 * tmp_1
    tmp_4 = tmp_3 * tmp_1

    tl.store(out_ptr + offsets, tmp_4, mask=mask)


@torch.fx.wrap
def fused_masked_fill(in_0):
    N = in_0.numel()
    out = torch.empty(in_0.shape, dtype=torch.float32, device=in_0.device)
    # BLOCK_SIZE=512 → single CUDA block; for N=484 both are < 512, so no extra warps
    BLOCK_SIZE = 512
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_masked_fill_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return fused_masked_fill