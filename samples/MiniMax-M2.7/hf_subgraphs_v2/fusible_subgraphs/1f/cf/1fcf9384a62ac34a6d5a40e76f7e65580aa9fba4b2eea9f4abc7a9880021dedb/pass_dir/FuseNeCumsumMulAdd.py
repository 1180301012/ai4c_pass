import torch
import triton
import triton.language as tl


@triton.jit
def fused_cumsum_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    seq_len: tl.constexpr,
):
    """
    Fused kernel for computing position indices from token IDs.
    
    Operations:
    - ne(1) -> int() -> cumsum(dim=1) -> type_as -> +0 -> *mask -> .long() -> +1
    
    One thread per batch element, sequential cumsum within each thread.
    """
    batch_idx = tl.program_id(0)
    
    # Sequential cumsum for this row
    cumsum_val = 0
    
    for col in range(seq_len):
        offset = batch_idx * seq_len + col
        mask = offset < n_elements
        
        # Load input token
        token = tl.load(in_ptr + offset, mask=mask, other=0)
        
        # tmp_1 = ne(1)
        mask_val = tl.where(token != 1, 1, 0)
        
        # tmp_2 = int() - implicit in mask_val
        
        # tmp_3 = cumsum(mask_val, dim=1) - sequential
        cumsum_val = cumsum_val + mask_val
        
        # tmp_6 = cumsum * mask_val
        result = cumsum_val * mask_val
        
        # tmp_7 = long(), tmp_8 = +1
        result = tl.cast(result, tl.int64) + 1
        
        # Store result
        tl.store(out_ptr + offset, result, mask=mask)


@torch.fx.wrap
def fused_cumsum_position(x: torch.Tensor) -> torch.Tensor:
    """
    Fused implementation using only tensor allocation APIs.
    """
    B, S = x.shape
    n_elements = B * S
    
    # Only use tensor allocation APIs
    out = torch.empty((B, S), dtype=torch.int64, device=x.device)
    
    # One program per batch element
    grid = (B,)
    
    fused_cumsum_kernel[grid](
        x,
        out,
        n_elements,
        S,
    )
    
    return out


def pattern(in_0):
    """
    Match the original computation pattern.
    This pattern produces position indices from token IDs.
    """
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_cumsum_position