import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    indices = in_0[0]
    result = in_1.index_select(-2, indices)
    return result


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def gather_kernel_2d(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    N_indices,
    N_features: tl.constexpr,
    in_0_row_stride,
    in_1_row_stride,
    out_row_stride,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    idx_start = pid * BLOCK_N
    idx_offsets = idx_start + tl.arange(0, BLOCK_N)
    idx_mask = idx_offsets < N_indices

    feat_offsets = tl.arange(0, N_features)

    # Load gather indices from row 0 of in_0 directly
    gather_indices = tl.load(in_0_ptr + idx_offsets, mask=idx_mask, other=0, eviction_policy='evict_first').to(tl.int32)

    # Gather rows from in_1 using indices
    src_offsets = gather_indices[:, None] * in_1_row_stride + feat_offsets[None, :]
    src_mask = idx_mask[:, None]
    values = tl.load(in_1_ptr + src_offsets, mask=src_mask, other=0.0, eviction_policy='evict_last')

    # Store to output
    dst_offsets = idx_offsets[:, None] * out_row_stride + feat_offsets[None, :]
    dst_mask = idx_mask[:, None]
    tl.store(out_ptr + dst_offsets, values, mask=dst_mask)


@triton.jit
def gather_kernel_1d(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    N_total,
    N_features: tl.constexpr,
    in_0_row_stride,
    in_1_row_stride,
    out_row_stride,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N_total

    # Convert flat offset to 2D
    row_idx = offsets // N_features
    col_idx = offsets % N_features

    # Load gather index from in_0 row 0
    gather_idx = tl.load(in_0_ptr + row_idx, mask=mask, other=0, eviction_policy='evict_first').to(tl.int32)

    # Load from source
    src_offset = gather_idx * in_1_row_stride + col_idx
    value = tl.load(in_1_ptr + src_offset, mask=mask, other=0.0, eviction_policy='evict_last')

    # Store to output
    dst_offset = row_idx * out_row_stride + col_idx
    tl.store(out_ptr + dst_offset, value, mask=mask)


@torch.fx.wrap
def fused_index_select(in_0, in_1):
    N_indices = in_0.shape[1]
    N_features = in_1.shape[1]

    out = torch.empty(N_indices, N_features, dtype=in_1.dtype, device=in_1.device)

    # Use 2D kernel with BLOCK_N=32 (17 programs for 1100 indices)
    BLOCK_N = 32
    grid = (triton.cdiv(N_indices, BLOCK_N),)

    gather_kernel_2d[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        N_indices=N_indices,
        N_features=N_features,
        in_0_row_stride=in_0.stride(0),
        in_1_row_stride=in_1.stride(0),
        out_row_stride=out.stride(0),
        BLOCK_N=BLOCK_N,
    )

    return out


def replacement_func():
    return fused_index_select