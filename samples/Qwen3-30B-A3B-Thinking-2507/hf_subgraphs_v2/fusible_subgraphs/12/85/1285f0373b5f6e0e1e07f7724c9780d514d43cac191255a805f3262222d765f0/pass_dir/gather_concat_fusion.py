import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches: tmp_1 = in_0[:, in_2]
#         tmp_9 = torch.cat([tmp_1, in_1], dim=1)
def pattern(in_0, in_1, in_2):
    tmp_1 = in_0[slice(None, None, None), in_2]
    tmp_9 = torch.cat([tmp_1, in_1], dim=1)
    return tmp_9

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Triton kernel to compute count of True in mask
@triton.jit
def count_true_kernel(mask_ptr, mask_size, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < mask_size
    values = tl.load(mask_ptr + offsets, mask=mask)
    count = tl.sum(values, axis=0)
    tl.store(output_ptr + pid, count)


# Triton kernel for direct gather + concat
@triton.jit
def gather_concat_kernel(
    in0_ptr,
    in1_ptr,
    mask_ptr,
    M,
    N,
    out_ptr,
    mask_size,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread processes a mask index
    mask_idx = tl.program_id(0)
    if mask_idx >= mask_size:
        return

    # Check if mask is True for this index
    mask_val = tl.load(mask_ptr + mask_idx)
    if not mask_val:
        return

    # Compute output column index (cumulative count of True)
    count = 0
    for i in range(mask_idx):
        if tl.load(mask_ptr + i):
            count += 1

    # Copy values from in0 to output
    for row in range(2):
        in0_val = tl.load(in0_ptr + row * mask_size + mask_idx)
        tl.store(out_ptr + row * (M + N) + count, in0_val)


# Wrapper function for the optimized concatenation
@torch.fx.wrap

def optimized_concat(in_0, in_1, in_2):
    # Compute M (number of True in mask) with Triton kernel
    mask_size = in_2.shape[0]
    M = torch.empty(1, device=in_0.device, dtype=torch.int32)
    count_true_kernel[(1,)](
        mask_ptr=in_2,
        mask_size=mask_size,
        output_ptr=M,
        BLOCK_SIZE=256
    )
    M = M.item()
    N = in_1.shape[1]

    # Preallocate output tensor
    output = torch.empty((2, M + N), dtype=in_0.dtype, device=in_0.device)

    # Launch gather+concat kernel
    gather_concat_kernel[(mask_size,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        mask_ptr=in_2,
        M=M,
        N=N,
        out_ptr=output,
        mask_size=mask_size,
        BLOCK_SIZE=256
    )

    # Copy in_1 to the remaining columns
    output[:, M:] = in_1
    return output

def replacement_func():
    return optimized_concat