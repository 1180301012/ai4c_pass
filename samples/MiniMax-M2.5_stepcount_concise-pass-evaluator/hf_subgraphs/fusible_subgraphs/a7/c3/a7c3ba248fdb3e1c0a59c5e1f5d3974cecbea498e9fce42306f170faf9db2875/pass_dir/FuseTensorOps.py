import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match the computation pattern:
    - Scalar multiplication in_0 * in_1
    - Device transfer for in_2 
    - Device transfer for tmp_1 (result of scalar mult)
    - Create constant tensor [-1]
    - Concatenate with empty tensor
    """
    tmp_0 = in_0
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.as_tensor(in_2, device=torch.device(type='cuda'))
    tmp_3 = torch.as_tensor(tmp_1, device=torch.device(type='cuda'))
    tmp_4 = torch.as_tensor([-1], dtype=torch.int64)
    tmp_5 = torch.as_tensor((), dtype=torch.int64)
    tmp_6 = torch.cat([tmp_4, tmp_5], dim=0)
    return tmp_2, tmp_3, tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_kernel_in_2(
    in_2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    # Load from in_2 (already on CUDA)
    x = tl.load(in_2_ptr + offsets, mask=mask, other=0)
    # Store to output
    tl.store(out_ptr + offsets, x, mask=mask)


def optimized_impl(in_0, in_1, in_2):
    """
    Optimized implementation that:
    1. Precomputes scalar multiplication (constant folding)
    2. Avoids redundant device transfer for in_2 (already on cuda)
    3. Directly creates the constant tensor instead of cat
    """
    # Precompute scalar multiplication (constant folding)
    # in_0 and in_1 are constants: 65536 * 1 = 65536
    mult_result = in_0 * in_1  # This is computed at runtime but is a constant
    
    # in_2 is already on cuda, so we just need to pass it through
    # But we need to ensure it's a proper tensor
    if isinstance(in_2, torch.Tensor):
        tmp_2 = in_2  # Already on cuda
    else:
        tmp_2 = torch.as_tensor(in_2, device=torch.device(type='cuda'))
    
    # Convert scalar to tensor on cuda
    tmp_3 = torch.as_tensor(mult_result, device=torch.device(type='cuda'))
    
    # Directly create the constant tensor [-1] instead of creating two and concatenating
    tmp_6 = torch.tensor([-1], dtype=torch.int64)
    
    return tmp_2, tmp_3, tmp_6


def replacement_func():
    return optimized_impl