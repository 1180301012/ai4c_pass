import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0, in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 1)
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    tmp_7 = tmp_6.to(device(type='cuda', index=0))
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return (tmp_13, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_cumsum_mask_expand_max_kernel(
    mask_ptr, data_ptr, out_max_ptr, out_exp_ptr,
    n_rows, n_cols,
    stride_mask_r, stride_mask_c,
    stride_data_r, stride_data_c,
    stride_out_max_r, stride_out_max_c,
    stride_out_exp_0, stride_out_exp_r, stride_out_exp_c,
):
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return

    cumsum = 0
    max_val = -1000000000

    for col in range(n_cols):
        # Load data value (in_1) and accumulate cumsum
        d = tl.load(data_ptr + row_id * stride_data_r + col * stride_data_c)
        cumsum = cumsum + d

        # Compute val = cumsum - 1 (tmp_2)
        val = cumsum - 1

        # Load mask (in_0) and apply masked_fill: where mask==0, set val=1
        m = tl.load(mask_ptr + row_id * stride_mask_r + col * stride_mask_c)
        if m == 0:
            val = 1

        # Track row-wise max
        if val > max_val:
            max_val = val

        # Store to expanded output (3 identical copies for tmp_7)
        tl.store(out_exp_ptr + 0 * stride_out_exp_0 + row_id * stride_out_exp_r + col * stride_out_exp_c, val)
        tl.store(out_exp_ptr + 1 * stride_out_exp_0 + row_id * stride_out_exp_r + col * stride_out_exp_c, val)
        tl.store(out_exp_ptr + 2 * stride_out_exp_0 + row_id * stride_out_exp_r + col * stride_out_exp_c, val)

    # Store max result: tmp_13 = max_val + 1 - 9 = max_val - 8
    result = max_val - 8
    tl.store(out_max_ptr + row_id * stride_out_max_r + 0 * stride_out_max_c, result)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    n_rows = in_0.shape[0]
    n_cols = in_0.shape[1]

    # tmp_13 shape: [n_rows, 1], dtype int64
    out_max = torch.empty((n_rows, 1), dtype=torch.int64, device=in_0.device)
    # tmp_7 shape: [3, n_rows, n_cols], dtype int64
    out_exp = torch.empty((3, n_rows, n_cols), dtype=torch.int64, device=in_0.device)

    grid = (n_rows,)

    fused_cumsum_mask_expand_max_kernel[grid](
        mask_ptr=in_0, data_ptr=in_1, out_max_ptr=out_max, out_exp_ptr=out_exp,
        n_rows=n_rows, n_cols=n_cols,
        stride_mask_r=in_0.stride(0), stride_mask_c=in_0.stride(1),
        stride_data_r=in_1.stride(0), stride_data_c=in_1.stride(1),
        stride_out_max_r=out_max.stride(0), stride_out_max_c=out_max.stride(1),
        stride_out_exp_0=out_exp.stride(0), stride_out_exp_r=out_exp.stride(1), stride_out_exp_c=out_exp.stride(2),
    )

    return (out_max, out_exp)


def replacement_func():
    return kernel_wrapper