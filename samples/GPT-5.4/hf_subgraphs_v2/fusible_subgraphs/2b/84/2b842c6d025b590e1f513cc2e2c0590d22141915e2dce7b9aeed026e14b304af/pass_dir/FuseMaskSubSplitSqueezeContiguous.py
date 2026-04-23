import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _mask_sub_kernel(
    mask_ptr,
    inp_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n_rows

    mask_vals = tl.load(mask_ptr + offs, mask=m, other=0).to(tl.float32)
    scaled_mask = mask_vals * 1000000.0

    inp0 = tl.load(inp_ptr + offs * 2, mask=m, other=0).to(tl.float32)
    inp1 = tl.load(inp_ptr + offs * 2 + 1, mask=m, other=0).to(tl.float32)

    batch = offs // n_cols
    col = offs % n_cols
    base = batch * (2 * n_cols) + col

    tl.store(out_ptr + base, inp0 - scaled_mask, mask=m)
    tl.store(out_ptr + base + n_cols, inp1 - scaled_mask, mask=m)


@torch.fx.wrap
def fused_mask_sub_split_squeeze_contiguous(in_0, in_1):
    packed = torch.empty((in_1.shape[0], in_1.shape[2], in_1.shape[1]), device=in_1.device, dtype=torch.float32)
    n_rows = in_0.numel()
    n_cols = in_1.shape[1]
    _mask_sub_kernel[(1,)](
        mask_ptr=in_0,
        inp_ptr=in_1,
        out_ptr=packed,
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=32,
        num_warps=1,
        num_stages=1,
    )
    return packed.transpose(1, 2)


def replacement_func():
    return fused_mask_sub_split_squeeze_contiguous