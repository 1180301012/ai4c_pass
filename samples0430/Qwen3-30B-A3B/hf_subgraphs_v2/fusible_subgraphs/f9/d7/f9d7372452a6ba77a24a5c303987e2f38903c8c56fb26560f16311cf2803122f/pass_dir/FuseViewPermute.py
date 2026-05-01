import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_3 = in_1.view(1, 32, -1)
    tmp_4 = tmp_3.permute(0, 2, 1)
    return tmp_4

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def reshape_permute_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Input shape: [1, 32, 64, 48] → flattened size = 98304
    # Output shape: [1, 64*48, 32] = [1, 3072, 32]
    for i in range(0, BLOCK_SIZE):
        if offsets[i] >= n_elements:
            break
        
        # Convert flattened index to input coordinates
        # h = head index, i_idx = height, j = width
        idx = offsets[i]
        h = (idx // (64 * 48)) % 32
        i_idx = (idx // 48) % 64
        j = idx % 48
        
        # Compute output index: (i_idx*48 + j)*32 + h
        out_idx = (i_idx * 48 + j) * 32 + h
        
        val = tl.load(in_ptr + idx, mask=mask)
        tl.store(out_ptr + out_idx, val, mask=mask)

@torch.fx.wrap
def fused_view_permute(in_1):
    n = in_1.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty(1, 64*48, 32, dtype=in_1.dtype, device=in_1.device)
    reshape_permute_kernel[(num_blocks,)](
        in_ptr=in_1,
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return fused_view_permute