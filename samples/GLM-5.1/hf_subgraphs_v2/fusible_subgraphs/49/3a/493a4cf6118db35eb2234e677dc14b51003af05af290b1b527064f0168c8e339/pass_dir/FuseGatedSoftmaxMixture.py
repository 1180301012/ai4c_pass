import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_2.softmax(dim=-1)
    tmp_2 = in_0.view(1, -1, 1, 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = tmp_6 * tmp_1
    tmp_8 = tmp_5 + tmp_7
    return tmp_8


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def gated_softmax_mixture_kernel_2d(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    n_cols,
    n_rows,
    n_channels,
    stride_c0,
    stride_b1, stride_c1, stride_r1, stride_col1,
    stride_b2, stride_c2, stride_r2, stride_col2,
    stride_b_out, stride_c_out, stride_r_out, stride_col_out,
    BLOCK_COL: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    # 2D grid: each program handles a block of rows and columns for one channel/batch
    pid_row = tl.program_id(0)
    pid_chan = tl.program_id(1)
    pid_batch = tl.program_id(2)

    row_start = pid_row * BLOCK_ROW
    row_offsets = row_start + tl.arange(0, BLOCK_ROW)
    row_mask = row_offsets < n_rows

    col_offsets = tl.arange(0, BLOCK_COL)
    col_mask = col_offsets < n_cols

    # Load gating parameter for this channel
    gating_val = tl.load(in_0_ptr + pid_chan * stride_c0).to(tl.float32)
    sig_val = tl.sigmoid(gating_val)
    one_m_sig = 1.0 - sig_val

    # 2D offsets for loading data
    row_2d = row_offsets[:, None]  # (BLOCK_ROW, 1)
    col_2d = col_offsets[None, :]  # (1, BLOCK_COL)
    mask_2d = row_mask[:, None] & col_mask[None, :]

    # Compute pointers for in_2 and compute softmax per row
    in_2_ptrs = in_2_ptr + pid_batch * stride_b2 + pid_chan * stride_c2 + row_2d * stride_r2 + col_2d * stride_col2
    in_2_block = tl.load(in_2_ptrs, mask=mask_2d, other=float('-inf')).to(tl.float32)

    # Softmax: subtract max, exp, sum, divide - per row
    row_max = tl.max(in_2_block, axis=1)[:, None]  # (BLOCK_ROW, 1)
    safe_block = in_2_block - row_max
    exp_block = tl.exp(safe_block)
    row_sum = tl.sum(exp_block, axis=1)[:, None]  # (BLOCK_ROW, 1)
    softmax_block = exp_block / row_sum

    # Load in_1 block
    in_1_ptrs = in_1_ptr + pid_batch * stride_b1 + pid_chan * stride_c1 + row_2d * stride_r1 + col_2d * stride_col1
    in_1_block = tl.load(in_1_ptrs, mask=mask_2d, other=0.0).to(tl.float32)

    # Compute gated mixture
    out_block = one_m_sig * in_1_block + sig_val * softmax_block

    # Store output
    out_ptrs = out_ptr + pid_batch * stride_b_out + pid_chan * stride_c_out + row_2d * stride_r_out + col_2d * stride_col_out
    tl.store(out_ptrs, out_block, mask=mask_2d)


@torch.fx.wrap
def gated_softmax_mixture(in_0, in_1, in_2):
    # Ensure in_0 is on the same device as in_1
    in_0 = torch.as_tensor(in_0, device=in_1.device)

    # Create output tensor
    out = torch.empty_like(in_1)

    n_cols = in_1.shape[-1]
    batch_size = in_1.shape[0]
    n_channels = in_0.shape[0]
    n_rows = in_1.shape[2]

    BLOCK_COL = triton.next_power_of_2(n_cols)
    BLOCK_ROW = 4  # Process multiple rows per program

    grid = ((n_rows // BLOCK_ROW) if n_rows % BLOCK_ROW == 0 else (n_rows // BLOCK_ROW + 1), n_channels, batch_size)

    gated_softmax_mixture_kernel_2d[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        n_cols=n_cols,
        n_rows=n_rows,
        n_channels=n_channels,
        stride_c0=in_0.stride(0),
        stride_b1=in_1.stride(0), stride_c1=in_1.stride(1),
        stride_r1=in_1.stride(2), stride_col1=in_1.stride(3),
        stride_b2=in_2.stride(0), stride_c2=in_2.stride(1),
        stride_r2=in_2.stride(2), stride_col2=in_2.stride(3),
        stride_b_out=out.stride(0), stride_c_out=out.stride(1),
        stride_r_out=out.stride(2), stride_col_out=out.stride(3),
        BLOCK_COL=BLOCK_COL,
        BLOCK_ROW=BLOCK_ROW,
    )

    return out


def replacement_func():
    return gated_softmax_mixture