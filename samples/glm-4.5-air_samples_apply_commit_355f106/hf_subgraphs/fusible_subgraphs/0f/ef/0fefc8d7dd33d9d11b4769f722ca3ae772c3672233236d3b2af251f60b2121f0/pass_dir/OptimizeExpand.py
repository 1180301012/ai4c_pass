import torch
import triton
import triton.language as tl

def pattern(x):
    return x.expand(1, -1, -1)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_4,)

@triton.jit
def expand_kernel(
    token_ptr, output_ptr,
    target_seq_len, n_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (target_seq_len * n_features)
    
    # Load the token value (it's the same for all positions)
    token_value = tl.load(token_ptr)
    
    # Calculate the output offset
    seq_pos = offsets // n_features
    feat_pos = offsets % n_features
    
    # Store the expanded token
    output_offset = seq_pos * n_features + feat_pos
    if mask:
        tl.store(output_ptr + output_offset, token_value)

@torch.fx.wrap
def optimized_expand(cls_token, target_seq_len):
    n_features = cls_token.shape[-1]
    total_elements = target_seq_len * n_features
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor on the same device as cls_token
    output = torch.empty((1, target_seq_len, n_features), 
                       dtype=cls_token.dtype, 
                       device=cls_token.device)
    
    # Flatten the cls_token for loading
    token_flat = cls_token.flatten()
    
    expand_kernel[(num_programs,)](
        token_flat, output,
        target_seq_len, n_features,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_expand