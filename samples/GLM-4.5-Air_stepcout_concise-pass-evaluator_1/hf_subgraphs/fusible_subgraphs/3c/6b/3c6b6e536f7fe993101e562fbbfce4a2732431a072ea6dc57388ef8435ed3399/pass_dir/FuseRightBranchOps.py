import torch
import triton
import triton.language as tl

# Pattern matching: Match the right branch operations
# -in_6 -> torch.cat((tmp_0, in_5), dim=-1) -> tmp_1 * in_2 -> tmp_3 + in_4 -> .to(dtype=torch.float32)
def pattern(in_5, in_6, in_2, in_4):
    tmp_0 = -in_6
    tmp_1 = torch.cat((tmp_0, in_5), dim=-1)
    tmp_2 = tmp_1 * in_2
    tmp_3 = in_4 + tmp_2
    tmp_4 = tmp_3.to(dtype=torch.float32)
    return tmp_4

def replacement_args(in_5, in_6, in_2, in_4):
    return (in_5, in_6, in_2, in_4)


@torch.fx.wrap
def fused_right_branch_kernel_wrapper(in_5, in_6, in_2, in_4):
    # Original computation:
    # tmp_0 = -in_6
    # tmp_1 = torch.cat((tmp_0, in_5), dim=-1)  # Shape: [B, H, L, 64]
    # tmp_2 = tmp_1 * in_2                       # Shape: [B, H, L, 64]
    # tmp_3 = in_4 + tmp_2                       # Shape: [B, H, L, 64]
    # tmp_4 = tmp_3.to(dtype=torch.float32)     # Shape: [B, H, L, 64]
    
    # Optimized: compute directly without intermediate tensors
    # Use in-place operations and avoid creating tmp_1
    
    # Compute negated in_6
    neg_in_6 = -in_6
    
    # Instead of concatenating, compute the result directly by 
    # splitting the computation along the last dimension
    # Result = (in_4 + (cat(-in_6, in_5) * in_2)).to(float32)
    
    # Get the half size (last dimension is 64, split into 32 + 32)
    last_dim = in_2.shape[-1]
    half_dim = last_dim // 2
    
    # Compute first half: in_4[..., :32] + (-in_6 * in_2[..., :32])
    first_half = in_4[..., :half_dim] + (neg_in_6 * in_2[..., :half_dim])
    
    # Compute second half: in_4[..., 32:] + (in_5 * in_2[..., 32:])
    second_half = in_4[..., half_dim:] + (in_5 * in_2[..., half_dim:])
    
    # Concatenate the halves
    output = torch.cat([first_half, second_half], dim=-1)
    
    return output.to(dtype=torch.float32)


def replacement_func():
    return fused_right_branch_kernel_wrapper