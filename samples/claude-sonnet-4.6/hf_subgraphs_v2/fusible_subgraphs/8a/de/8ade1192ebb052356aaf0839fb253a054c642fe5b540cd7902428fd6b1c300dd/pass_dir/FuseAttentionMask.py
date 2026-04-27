import torch
import triton
import triton.language as tl


def pattern(in_5):
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = tmp_5 - tmp_4
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8


def replacement_args(in_5):
    return (in_5,)


@triton.jit
def attention_mask_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load int64 values
    x = tl.load(in_ptr + offsets, mask=mask, other=1)
    # Logic: val = 1.0 - float(x); bool(val) is True when val != 0 (i.e. x != 1)
    # masked_fill: output = -3.4e38 where bool is True, else val (= 0.0 when x==1)
    out = tl.where(x == 1, 0.0, -3.4028234663852886e+38).to(tl.float32)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_attention_mask(in_5):
    n = in_5.numel()
    BLOCK_SIZE = 256
    out = torch.empty(in_5.shape, dtype=torch.float32, device=in_5.device)
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    attention_mask_kernel[grid](
        in_5,
        out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replacement_func():
    return fused_attention_mask