import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match the exact computation pattern from the float16/3 example"""
    # in_1 += in_0; in_2 = in_1
    in_1 += in_0
    in_2 = in_1
    
    # tmp_1 = in_2.float()
    tmp_1 = in_2.float()
    
    # tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    
    # tmp_3 = tmp_2.type_as(in_2)
    tmp_3 = tmp_2.type_as(in_2)
    
    # tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    
    return (tmp_4,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def float16_optimization_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for the computation pattern"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Element-wise addition
    added = in_1 + in_0
    
    # Get max for numerical stability
    max_val = tl.max(added, mask=mask)
    shifted = added - tl.broadcast_to(max_val, added.shape)
    
    # Exponentiate
    exp_val = tl.exp(shifted, mask=mask)
    exp_sum = tl.sum(exp_val, mask=mask)
    exp_sum = tl.broadcast_to(exp_sum, exp_val.shape)
    
    # Softmax
    softmax_result = exp_val / exp_sum
    
    # Store result (dropout not applied during inference)
    tl.store(out_ptr + offsets, softmax_result, mask=mask)

@torch.fx.wrap
def float16_optimized_func(in_0, in_1):
    """Function that implements the optimized computation"""
    # Flattened tensors for processing
    in_0_flat = in_0.reshape(-1)
    in_1_flat = in_1.reshape(-1)
    out = torch.empty_like(in_0)
    out_flat = out.reshape(-1)
    
    N = in_0_flat.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    float16_optimization_kernel[(num_programs,)](
        in_0_ptr=in_0_flat,
        in_1_ptr=in_1_flat,
        out_ptr=out_flat,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return float16_optimized_func