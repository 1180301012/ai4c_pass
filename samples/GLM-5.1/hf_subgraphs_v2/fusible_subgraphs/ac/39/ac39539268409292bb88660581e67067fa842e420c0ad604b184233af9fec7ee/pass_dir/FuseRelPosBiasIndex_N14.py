import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    return tmp_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def triton_cat_kernel(
    in1_ptr, in0_ptr, out_ptr,
    M1, M0, K,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < K

    if row_idx < M1:
        src_ptr = in1_ptr + row_idx * K + col_offsets
    else:
        src_ptr = in0_ptr + (row_idx - M1) * K + col_offsets

    dst_ptr = out_ptr + row_idx * K + col_offsets
    data = tl.load(src_ptr, mask=col_mask, other=0.0)
    tl.store(dst_ptr, data, mask=col_mask)


@torch.fx.wrap
def triton_cat(in_0, in_1):
    M1 = in_1.shape[0]
    M0 = in_0.shape[0]
    K = in_1.shape[1]

    cat_out = torch.empty((M1 + M0, K), dtype=in_1.dtype, device=in_1.device)
    BLOCK_SIZE = 16
    num_col_programs = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    triton_cat_kernel[(M1 + M0, num_col_programs)](
        in1_ptr=in_1, in0_ptr=in_0, out_ptr=cat_out,
        M1=M1, M0=M0, K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return cat_out


def replacement_func():
    return triton_cat