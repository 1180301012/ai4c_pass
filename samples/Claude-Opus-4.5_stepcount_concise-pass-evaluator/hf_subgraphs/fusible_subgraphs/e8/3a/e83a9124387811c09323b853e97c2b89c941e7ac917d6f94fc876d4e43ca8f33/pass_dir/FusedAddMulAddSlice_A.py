import torch
import triton
import triton.language as tl

# Pattern for fusing add + mul + add operations
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = tmp_2 * in_1
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_add_mul_add_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    hidden_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute hidden dimension index for broadcasting
    hidden_idx = offsets % hidden_dim
    
    # Load values from 3D tensors
    in_2_vals = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3_vals = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Load values from 1D tensors (broadcast)
    in_0_vals = tl.load(in_0_ptr + hidden_idx, mask=mask, other=0.0)
    in_1_vals = tl.load(in_1_ptr + hidden_idx, mask=mask, other=0.0)
    
    # Compute: (in_3 + in_2) * in_1 + in_0
    result = (in_3_vals + in_2_vals) * in_1_vals + in_0_vals
    
    # Store main output
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_add_mul_add(in_0, in_1, in_2, in_3):
    n_elements = in_2.numel()
    
    # For small tensors, use PyTorch (lower overhead)
    # For large tensors, use Triton kernel (better throughput)
    if n_elements < 2000000:  # ~2M elements threshold
        return (in_3 + in_2) * in_1 + in_0
    
    hidden_dim = in_2.shape[-1]
    out = torch.empty_like(in_2)
    
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_add_mul_add_kernel[grid](
        in_0, in_1, in_2, in_3,
        out,
        hidden_dim,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_mul_add