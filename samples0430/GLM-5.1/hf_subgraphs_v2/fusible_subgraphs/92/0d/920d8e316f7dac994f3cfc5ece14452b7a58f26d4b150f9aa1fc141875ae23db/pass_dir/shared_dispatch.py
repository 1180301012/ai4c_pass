import torch
import triton
import triton.language as tl


@triton.jit
def arange_view_repeat_kernel(
    out_ptr,
    N,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Full fusion kernel: directly computes arange values repeated.
    Each element at flat index pid has value (pid % N).
    """
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < n_elements
    col = pid % N
    # Cast to output dtype
    tl.store(out_ptr + pid, col, mask=mask)


@triton.jit
def view_repeat_kernel(
    out_ptr,
    in_ptr,
    N,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Partial fusion kernel: reads from arange result and writes repeated output.
    Each element at flat index pid reads from in_ptr[pid % N].
    """
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < n_elements
    col = pid % N
    in_mask = col < N
    value = tl.load(in_ptr + col, mask=in_mask, other=0)
    tl.store(out_ptr + pid, value, mask=mask)


@torch.fx.wrap
def dispatch_wrapper(*args):
    """Shared dispatch wrapper for all passes.
    Routes based on the last argument (route string).
    Returns a TENSOR (not a tuple) to match the pattern's returning node.
    """
    route = args[-1]
    
    if route == "full_128":
        N = 128
        repeat_dim0 = 2
        out = torch.empty((repeat_dim0, N), dtype=torch.int64, device='cuda')
        n_elements = repeat_dim0 * N
        BLOCK_SIZE = 256
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        arange_view_repeat_kernel[grid](out, N, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out
    
    elif route == "full_1000":
        N = 1000
        repeat_dim0 = 2
        out = torch.empty((repeat_dim0, N), dtype=torch.int64, device='cuda')
        n_elements = repeat_dim0 * N
        BLOCK_SIZE = 256
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        arange_view_repeat_kernel[grid](out, N, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out
    
    elif route == "partial":
        arange_result = args[0]
        N = arange_result.shape[-1]
        repeat_dim0 = 2
        out = torch.empty((repeat_dim0, N), dtype=arange_result.dtype, device=arange_result.device)
        n_elements = repeat_dim0 * N
        BLOCK_SIZE = 256
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        view_repeat_kernel[grid](out, arange_result, N, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return out
    
    else:
        raise ValueError(f"Unknown route: {route}")