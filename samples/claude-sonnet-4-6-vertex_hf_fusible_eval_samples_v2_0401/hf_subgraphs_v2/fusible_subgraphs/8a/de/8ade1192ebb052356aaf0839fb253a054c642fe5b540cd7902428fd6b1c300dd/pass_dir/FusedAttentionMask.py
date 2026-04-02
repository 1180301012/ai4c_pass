import torch
import operator
import triton
import triton.language as tl


def pattern(in_5):
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = operator.sub(tmp_5, tmp_4)
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8


def replacement_args(in_5):
    return (in_5,)


@triton.jit
def fused_attention_mask_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load int64 values and convert to float32
    x = tl.load(in_ptr + offsets, mask=mask, other=1).to(tl.float32)
    # Compute 1.0 - x
    val = 1.0 - x
    # masked_fill: where val != 0 (bool True), fill with -3.4e38
    result = tl.where(val != 0.0, -3.4028234663852886e+38, val)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_attention_mask(in_5):
    n_elements = in_5.numel()
    out = torch.empty(in_5.shape, dtype=torch.float32, device=in_5.device)
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_attention_mask_kernel[grid](
        in_5,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return fused_attention_mask