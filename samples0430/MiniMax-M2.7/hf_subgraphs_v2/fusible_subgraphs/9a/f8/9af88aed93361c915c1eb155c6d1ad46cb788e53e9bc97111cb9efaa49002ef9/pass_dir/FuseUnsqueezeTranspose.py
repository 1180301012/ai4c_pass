import torch
import triton
import triton.language as tl


@triton.jit
def fused_unsqueeze_transpose_kernel(
    in_ptr,
    out_ptr,
    B: tl.int32,
    T: tl.int32,
    D: tl.int32,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for unsqueeze(1) + transpose(2, 3).
    
    Input shape: [B, T, D] -> unsqueeze(1) -> [B, 1, T, D] -> transpose(2,3) -> [B, 1, D, T]
    
    For output position [b, 0, d, t], we need input position [b, t, d].
    
    Uses vectorized loads/stores for better memory bandwidth.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute output coordinates from linear index
    # output[b, 0, d, t] where linear_idx = b*D*T + d*T + t
    # Precompute D*T to avoid repeated computation
    DT = D * T
    b = offsets // DT
    rem = offsets % DT
    d = rem // T
    t = rem % T
    
    # Compute input linear index for [b, t, d]: b*T*D + t*D + d
    TD = T * D
    input_offsets = b * TD + t * D + d
    
    # Load and store
    x = tl.load(in_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_unsqueeze_transpose_wrapper(in_0):
    """
    Wrapper that implements unsqueeze(1) followed by transpose(2,3).
    
    Input: [B, T, D] -> Output: [B, 1, D, T]
    """
    B, T, D = in_0.shape
    
    # Output shape after unsqueeze(1) + transpose(2,3): [B, 1, D, T]
    out = torch.empty((B, 1, D, T), dtype=in_0.dtype, device=in_0.device)
    
    n_elements = B * T * D
    
    # Use very large block size to minimize kernel launch overhead
    BLOCK_SIZE = 4096
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if num_programs > 0:
        fused_unsqueeze_transpose_kernel[(num_programs,)](
            in_ptr=in_0,
            out_ptr=out,
            B=B,
            T=T,
            D=D,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out


def pattern(in_0):
    """Match the unsqueeze(1) + transpose(2, 3) pattern."""
    tmp_1 = in_0.unsqueeze(1)
    tmp_2 = tmp_1.transpose(2, 3)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_unsqueeze_transpose_wrapper