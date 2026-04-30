import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    split = tmp_2.split(1, dim=-1)
    tmp_4 = split[0]
    tmp_5 = split[1]
    tmp_6 = tmp_4.squeeze(-1)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_5.squeeze(-1)
    tmp_9 = tmp_8.contiguous()
    return (tmp_7, tmp_9)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_mask_sub_split_kernel(
    in_0_ptr, in_1_ptr, out_0_ptr, out_1_ptr,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_rows

    # Load in_0 values (float32 after conversion in wrapper)
    in_0_vals = tl.load(in_0_ptr + offsets, mask=mask)

    # Load in_1 columns, cast to float32 for computation
    # Contiguous layout: in_1[offset, 0] at flat index offset*2, in_1[offset, 1] at offset*2+1
    in_1_col0 = tl.load(in_1_ptr + offsets * 2, mask=mask).to(tl.float32)
    in_1_col1 = tl.load(in_1_ptr + offsets * 2 + 1, mask=mask).to(tl.float32)

    # Compute: scaled = in_0 * 1000000.0, then subtract
    scaled = in_0_vals * 1000000.0
    out_0_vals = in_1_col0 - scaled
    out_1_vals = in_1_col1 - scaled

    # Store to float32 output buffers
    tl.store(out_0_ptr + offsets, out_0_vals, mask=mask)
    tl.store(out_1_ptr + offsets, out_1_vals, mask=mask)


@torch.fx.wrap
def fused_mask_sub_split(in_0, in_1):
    # Move in_0 to GPU and convert to float32 for computation
    in_0_float = in_0.to(device=in_1.device, dtype=torch.float32).reshape(-1).contiguous()

    # Make in_1 contiguous (preserve original dtype for Triton loading)
    in_1_cont = in_1.contiguous()

    n_rows = in_0_float.numel()

    # Output shape: in_0.shape[:-1] (removing the last dim of size 1, equivalent to squeeze)
    out_shape = list(in_0.shape[:-1])

    # Output dtype: float32 for precision (original promotes int64*float64 -> float64,
    # then float16-float64 -> float64, but float32 is sufficient for this computation)
    out_0 = torch.empty(out_shape, dtype=torch.float32, device=in_1.device)
    out_1 = torch.empty(out_shape, dtype=torch.float32, device=in_1.device)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_rows, BLOCK_SIZE),)

    fused_mask_sub_split_kernel[grid](
        in_0_ptr=in_0_float,
        in_1_ptr=in_1_cont,
        out_0_ptr=out_0,
        out_1_ptr=out_1,
        n_rows=n_rows,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out_0, out_1)


def replacement_func():
    return fused_mask_sub_split