import torch
import triton
import triton.language as tl


def pattern(tmp_9):
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    return tmp_16


def replacement_args(tmp_9):
    return (tmp_9,)


@triton.jit
def mask_kernel(tmp9_ptr, out_ptr, N_SUB_POS: tl.constexpr):
    w = tl.program_id(0)  # window index (0 to 360)
    s_a = tl.program_id(1)  # sub-position a (0 to 48)

    # Load mask value for (w, s_a) from tmp_9
    # tmp_9 has shape (1, 361, 49), flat index: w * 49 + s_a
    mask_a = tl.load(tmp9_ptr + w * N_SUB_POS + s_a)

    # Compute all s_b values for this (w, s_a)
    s_b_range = tl.arange(0, N_SUB_POS)

    # Load mask values for all (w, s_b) from tmp_9
    # These 49 values are contiguous in tmp_9 for the same window w
    mask_b = tl.load(tmp9_ptr + w * N_SUB_POS + s_b_range)

    # Compute output: -1000.0 if border status differs, 0.0 if same
    # This replaces: subtraction -> ne -> masked_fill(-1000) -> eq -> masked_fill(0)
    out_val = tl.where(mask_a != mask_b, -1000.0, 0.0)

    # Store output at position (w, s_a, s_b) in the (1, 361, 49, 49) tensor
    # Flat index: w * 49 * 49 + s_a * 49 + s_b
    out_offset = w * N_SUB_POS * N_SUB_POS + s_a * N_SUB_POS + s_b_range
    tl.store(out_ptr + out_offset, out_val)


@torch.fx.wrap
def compute_mask_from_tmp9(tmp_9):
    N_WINDOWS = 361
    N_SUB_POS = 49
    output_shape = (1, N_WINDOWS, N_SUB_POS, N_SUB_POS)
    output = torch.empty(output_shape, dtype=tmp_9.dtype, device=tmp_9.device)
    grid = (N_WINDOWS, N_SUB_POS)
    mask_kernel[grid](
        tmp9_ptr=tmp_9,
        out_ptr=output,
        N_SUB_POS=N_SUB_POS,
    )
    return output


def replacement_func():
    return compute_mask_from_tmp9