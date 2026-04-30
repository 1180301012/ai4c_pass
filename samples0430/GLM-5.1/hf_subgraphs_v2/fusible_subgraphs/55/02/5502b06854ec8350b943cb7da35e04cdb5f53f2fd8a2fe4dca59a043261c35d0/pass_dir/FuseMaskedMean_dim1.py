import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_masked_mean_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    cols,
    stride_in0_b, stride_in0_r, stride_in0_c,
    stride_in1_b, stride_in1_r, stride_in1_c,
    stride_out_b, stride_out_c,
    BATCH: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    col_start = pid * BLOCK_COLS
    col_offsets = col_start + tl.arange(0, BLOCK_COLS)
    col_mask = col_offsets < cols

    for b in range(BATCH):
        sum_prod = tl.zeros([BLOCK_COLS], dtype=tl.float32)
        sum_mask = tl.zeros([BLOCK_COLS], dtype=tl.float32)

        for r in range(ROWS):
            offset_in0 = b * stride_in0_b + r * stride_in0_r + col_offsets * stride_in0_c
            mask_val = tl.load(in_0_ptr + offset_in0, mask=col_mask, other=0).to(tl.float32)

            offset_in1 = b * stride_in1_b + r * stride_in1_r + col_offsets * stride_in1_c
            data_val = tl.load(in_1_ptr + offset_in1, mask=col_mask, other=0.0).to(tl.float32)

            sum_prod += data_val * mask_val
            sum_mask += mask_val

        clamped_mask = tl.maximum(sum_mask, 1e-9)
        result = sum_prod / clamped_mask

        offset_out = b * stride_out_b + col_offsets * stride_out_c
        tl.store(out_ptr + offset_out, result, mask=col_mask)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    batch = in_0.shape[0]
    rows = in_0.shape[1]
    cols = in_0.shape[2]

    out = torch.empty((batch, cols), dtype=torch.float32, device=in_0.device)

    stride_in0 = in_0.stride()
    stride_in1 = in_1.stride()
    stride_out = out.stride()

    BLOCK_COLS = 1024
    grid = ((cols + BLOCK_COLS - 1) // BLOCK_COLS,)

    fused_masked_mean_kernel[grid](
        in_0, in_1, out,
        cols,
        stride_in0[0], stride_in0[1], stride_in0[2],
        stride_in1[0], stride_in1[1], stride_in1[2],
        stride_out[0], stride_out[1],
        BATCH=batch,
        ROWS=rows,
        BLOCK_COLS=BLOCK_COLS,
    )

    return out


def replacement_func():
    return kernel_wrapper