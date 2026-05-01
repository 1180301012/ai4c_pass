import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    result = in_2 * in_1 + in_0
    unbind = torch.unbind(result, dim=2)
    part0 = unbind[0]
    part1 = unbind[1]
    permuted = part1.permute(0, 2, 1)
    return permuted, part0

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_kernel(
    in_2_ptr,
    in_1_ptr,
    in_0_ptr,
    first_slice_ptr,
    second_permuted_ptr,
    N,
    M,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < K

    n = pid // M
    m = pid % M

    # Load from in_2 (shape [N, M, 1, K])
    in_2_ptr_base = in_2_ptr + n * M * K + m * K
    in_2 = tl.load(in_2_ptr_base + offsets, mask=mask, other=0.0)

    # Load in_1 (shape [1, 1, 2, K]) for channels 0 and 1
    in_1_ch0 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_1_ch1 = tl.load(in_1_ptr + K + offsets, mask=mask, other=0.0)

    # Load in_0 (shape [2, K])
    in_0_ch0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_0_ch1 = tl.load(in_0_ptr + K + offsets, mask=mask, other=0.0)

    # Compute both outputs
    first_slice_val = in_2 * in_1_ch0 + in_0_ch0
    second_permuted_val = in_2 * in_1_ch1 + in_0_ch1

    # Store first_slice (shape [N, M, K])
    first_slice_ptr_base = first_slice_ptr + n * M * K + m * K
    tl.store(first_slice_ptr_base + offsets, first_slice_val, mask=mask)

    # Store second_permuted (shape [N, K, M])
    second_permuted_ptr_base = second_permuted_ptr + n * K * M + offsets * M + m
    tl.store(second_permuted_ptr_base, second_permuted_val, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    N = in_2.shape[0]
    M = in_2.shape[1]
    K = in_2.shape[3]

    first_slice = torch.empty((N, M, K), dtype=in_2.dtype, device=in_2.device)
    second_permuted = torch.empty((N, K, M), dtype=in_2.dtype, device=in_2.device)

    BLOCK_SIZE = 128
    num_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_kernel[(num_blocks,)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        in_0_ptr=in_0,
        first_slice_ptr=first_slice,
        second_permuted_ptr=second_permuted,
        N=N,
        M=M,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return second_permuted, first_slice

def replacement_func():
    return fused_kernel_wrapper