import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_0[slice(None, None, None), in_2]
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    return (tmp_1, tmp_9)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['total_cat', 'K', 'M'],
)
@triton.jit
def fused_gather_cat_kernel(
    in_0_ptr, in_1_ptr, indices_ptr,
    out_index_ptr, out_cat_ptr,
    total_cat, total_index, K, M, R,
    stride_in0_0, stride_in0_1,
    stride_in1_0, stride_in1_1,
    stride_out_index_0, stride_out_index_1,
    stride_out_cat_0, stride_out_cat_1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Process out_cat elements
    cat_mask = offsets < total_cat
    total_cols = K + M
    cat_row = offsets // total_cols
    cat_col = offsets % total_cols

    is_indexed = cat_col < K

    # For indexed columns, load the source column index
    # Use other=0 as fallback for non-indexed columns (won't be used)
    src_col = tl.load(indices_ptr + cat_col, mask=cat_mask & is_indexed, other=0)

    # Compute safe offsets for in_0 data
    in0_offset = cat_row * stride_in0_0 + src_col * stride_in0_1
    # Compute safe offsets for in_1 data (use 0 column when indexed, won't be used)
    in1_col_safe = tl.where(is_indexed, 0, cat_col - K)
    in1_offset = cat_row * stride_in1_0 + in1_col_safe * stride_in1_1

    # Load values with appropriate masks
    in0_val = tl.load(in_0_ptr + in0_offset, mask=cat_mask & is_indexed, other=0)
    in1_val = tl.load(in_1_ptr + in1_offset, mask=cat_mask & ~is_indexed, other=0)

    # Select the right value
    cat_val = tl.where(is_indexed, in0_val, in1_val)

    # Store to out_cat
    out_cat_offset = cat_row * stride_out_cat_0 + cat_col * stride_out_cat_1
    tl.store(out_cat_ptr + out_cat_offset, cat_val, mask=cat_mask)

    # Process out_index elements (only the indexed portion)
    index_mask = offsets < total_index
    index_row = offsets // K if K > 0 else 0
    index_col = offsets % K if K > 0 else 0

    # Load source column index for out_index
    src_col_idx = tl.load(indices_ptr + index_col, mask=index_mask, other=0)

    # Load from in_0
    in0_offset_idx = index_row * stride_in0_0 + src_col_idx * stride_in0_1
    index_val = tl.load(in_0_ptr + in0_offset_idx, mask=index_mask, other=0)

    # Store to out_index
    out_index_offset = index_row * stride_out_index_0 + index_col * stride_out_index_1
    tl.store(out_index_ptr + out_index_offset, index_val, mask=index_mask)


@torch.fx.wrap
def fused_index_cat(in_0, in_1, in_2):
    # Move in_0 to CUDA if needed (using allowed allocation API)
    in_0_cuda = torch.as_tensor(in_0, device=in_1.device)

    # Get dimensions
    R = in_0_cuda.shape[0]
    N = in_0_cuda.shape[1]
    M = in_1.shape[1]

    # Count True values in mask
    K = in_2.sum().item()

    # Get indices of True values
    if K > 0:
        true_indices = in_2.nonzero().flatten().to(torch.int64)
    else:
        true_indices = torch.empty((0,), dtype=torch.int64, device=in_1.device)

    # Allocate outputs using allowed APIs
    out_index = torch.empty((R, K), dtype=in_0_cuda.dtype, device=in_0_cuda.device)
    out_cat = torch.empty((R, K + M), dtype=in_1.dtype, device=in_1.device)

    # Handle edge case: K + M = 0 (unlikely but safe)
    total_cat = R * (K + M)
    total_index = R * K

    if total_cat > 0:
        BLOCK_SIZE = 1024
        num_programs = (total_cat + BLOCK_SIZE - 1) // BLOCK_SIZE

        fused_gather_cat_kernel[(num_programs,)](
            in_0_ptr=in_0_cuda,
            in_1_ptr=in_1,
            indices_ptr=true_indices,
            out_index_ptr=out_index,
            out_cat_ptr=out_cat,
            total_cat=total_cat,
            total_index=total_index,
            K=K,
            M=M,
            R=R,
            stride_in0_0=in_0_cuda.stride(0),
            stride_in0_1=in_0_cuda.stride(1),
            stride_in1_0=in_1.stride(0),
            stride_in1_1=in_1.stride(1),
            stride_out_index_0=out_index.stride(0),
            stride_out_index_1=out_index.stride(1),
            stride_out_cat_0=out_cat.stride(0),
            stride_out_cat_1=out_cat.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return (out_index, out_cat)


def replacement_func():
    return fused_index_cat