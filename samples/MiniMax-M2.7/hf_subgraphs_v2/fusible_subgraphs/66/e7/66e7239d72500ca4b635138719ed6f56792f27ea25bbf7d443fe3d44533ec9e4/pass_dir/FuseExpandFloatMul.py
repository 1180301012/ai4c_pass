import torch
import triton
import triton.language as tl

# Pattern matching the exact model operations
def pattern(in_0, in_1, in_2, in_3):
    # Layer norm
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    # Unsqueeze
    tmp_5 = in_0.unsqueeze(-1)
    # Expand
    tmp_6 = tmp_5.expand_as(tmp_4)
    # Convert to float
    tmp_7 = tmp_6.to(torch.float32)
    # Multiply
    tmp_8 = torch.mul(tmp_4, tmp_7)
    return tmp_7, tmp_8, tmp_4

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_kernel(
    x_ptr, mask_ptr, out_mask_ptr, out_result_ptr,
    M, N, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N
    
    m_indices = offsets // N
    n_indices = offsets % N
    
    # Load x [M, N]
    x = tl.load(x_ptr + m_indices * N + n_indices, mask=mask, other=0.0)
    
    # Load mask [M, 1]
    mask_val = tl.load(mask_ptr + m_indices, mask=m_indices < M, other=0.0)
    
    # Broadcast and convert
    mask_float = mask_val.to(tl.float32)
    result = (x * mask_float).to(tl.float32)
    
    tl.store(out_mask_ptr + m_indices * N + n_indices, mask_float, mask=mask)
    tl.store(out_result_ptr + m_indices * N + n_indices, result, mask=mask)

@torch.fx.wrap
def fused_impl(x, mask, weight, bias):
    M, N = x.shape
    total = M * N
    BLOCK_SIZE = 1024
    num_programs = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out_mask = torch.empty((M, N), dtype=torch.float32, device=x.device)
    out_result = torch.empty((M, N), dtype=torch.float32, device=x.device)
    
    fused_kernel[(num_programs,)](
        x, mask, out_mask, out_result, M, N, BLOCK_SIZE,
    )
    
    return out_mask, out_result, x

def replacement_func():
    return fused_impl