import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # Match the addition pattern from the model
    tmp_0 = in_0 + in_3
    tmp_1 = tmp_0 + in_2
    tmp_2 = tmp_1 / 8.0
    tmp_3 = tmp_2 + in_1
    # Include all intermediate values as expected by pattern matching framework
    return tmp_0, tmp_1, tmp_2, tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_addition_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    tmp_0_ptr, tmp_1_ptr, tmp_2_ptr, tmp_3_ptr,
    batch_size, num_heads, seq_len,
    scale_factor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * num_heads * seq_len * seq_len
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load all inputs
    in_0_elem = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1_elem = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    in_2_elem = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    in_3_elem = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operations: tmp_3 = (in_0 + in_1 + in_2 + in_3) / 8.0
    result = (in_0_elem + in_1_elem + in_2_elem + in_3_elem) / scale_factor
    
    # Store all intermediate results (in practice, we could optimize this)
    tl.store(tmp_0_ptr + offsets, in_0_elem + in_3_elem, mask=mask)
    tl.store(tmp_1_ptr + offsets, result * scale_factor, mask=mask)  # tmp_1 = tmp_0 + in_2
    tl.store(tmp_2_ptr + offsets, result * scale_factor / scale_factor, mask=mask)  # tmp_2 = tmp_1 / 8.0  
    tl.store(tmp_3_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_addition_forward(in_0, in_1, in_2, in_3):
    batch_size, num_heads, seq_len, _ = in_0.shape
    total_elements = batch_size * num_heads * seq_len * seq_len
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create intermediate tensors
    tmp_0 = torch.empty_like(in_0)
    tmp_1 = torch.empty_like(in_0)
    tmp_2 = torch.empty_like(in_0)
    tmp_3 = torch.empty_like(in_0)
    
    # Flatten inputs for easier indexing
    in_0_flat = in_0.flatten()
    in_1_flat = in_1.flatten()
    in_2_flat = in_2.flatten()
    in_3_flat = in_3.flatten()
    tmp_0_flat = tmp_0.flatten()
    tmp_1_flat = tmp_1.flatten()
    tmp_2_flat = tmp_2.flatten()
    tmp_3_flat = tmp_3.flatten()
    
    # Launch kernel
    fused_addition_kernel[num_programs](
        in_0_ptr=in_0_flat,
        in_1_ptr=in_1_flat,
        in_2_ptr=in_2_flat,
        in_3_ptr=in_3_flat,
        tmp_0_ptr=tmp_0_flat,
        tmp_1_ptr=tmp_1_flat,
        tmp_2_ptr=tmp_2_flat,
        tmp_3_ptr=tmp_3_flat,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        scale_factor=8.0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return tmp_0, tmp_1, tmp_2, tmp_3

def replacement_func():
    return fused_addition_forward