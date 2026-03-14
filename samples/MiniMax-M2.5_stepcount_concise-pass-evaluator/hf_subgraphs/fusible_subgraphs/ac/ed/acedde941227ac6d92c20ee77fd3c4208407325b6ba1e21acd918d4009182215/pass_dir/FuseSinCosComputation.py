import torch
import triton
import triton.language as tl


def pattern(in_2):
    """
    Pattern to match: cat + cos + sin with multiply by 1.0 and type conversion
    The pattern matches the exact computation structure from the model.
    The outputs (tmp_7, tmp_8) are what appear in the model's return.
    """
    tmp_2 = torch.cat((in_2, in_2), dim=-1)
    tmp_3 = tmp_2.cos()
    tmp_4 = tmp_3 * 1.0
    tmp_5 = tmp_2.sin()
    tmp_6 = tmp_5 * 1.0
    tmp_7 = tmp_4.to(dtype=torch.float16)
    tmp_8 = tmp_6.to(dtype=torch.float16)
    # Return as stacked to match replacement
    stacked = torch.stack([tmp_7, tmp_8], dim=0)
    return stacked


def replacement_args(in_2):
    return (in_2,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_sin_cos_kernel(
    in_ptr,
    cos_out_ptr,
    sin_out_ptr,
    seq_len,
    orig_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # For tensor with shape [batch, seq_len, orig_dim], flattened row-major:
    # output has shape [batch, seq_len, 2*orig_dim] -> flattened has seq_len*2*orig_dim elements
    # 
    # For output index i:
    # - seq_idx = i // (2 * orig_dim)
    # - feat_idx = i % (2 * orig_dim)
    # - Input index = seq_idx * orig_dim + (feat_idx % orig_dim)
    # 
    # We compute: input_idx = (i // (2 * orig_dim)) * orig_dim + (i % (2 * orig_dim)) % orig_dim
    
    two_orig_dim = 2 * orig_dim
    seq_idx = offsets // two_orig_dim
    feat_idx = offsets % two_orig_dim
    orig_feat_idx = feat_idx % orig_dim
    orig_offsets = seq_idx * orig_dim + orig_feat_idx
    
    # Load input from the original tensor using the computed offset
    x = tl.load(in_ptr + orig_offsets, mask=mask, other=0.0)
    
    # Compute sin and cos
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    # Convert to float16 and store
    cos_val_fp16 = cos_val.to(tl.float16)
    sin_val_fp16 = sin_val.to(tl.float16)
    
    tl.store(cos_out_ptr + offsets, cos_val_fp16, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val_fp16, mask=mask)


@torch.fx.wrap
def fused_sin_cos_kernel_wrapper(in_2):
    """
    Fused kernel that:
    1. Handles the concatenation (doubles the last dimension by repeating)
    2. Computes both sin and cos in a single kernel launch
    3. Converts to float16
    4. Returns a stacked tensor [2, batch, seq_len, 32]
    """
    # Input shape: [1, seq_len, 16] -> After cat: [1, seq_len, 32]
    input_shape = in_2.shape  # [1, seq_len, 16]
    seq_len = input_shape[1]
    orig_dim = input_shape[2]  # 16
    concat_dim = orig_dim * 2  # 32
    
    # Total elements in the output (after concatenation)
    n_elements = seq_len * concat_dim
    
    # Allocate output tensors with correct shape [1, seq_len, 32]
    output_shape = (input_shape[0], seq_len, concat_dim)
    cos_out = torch.empty(output_shape, dtype=torch.float16, device=in_2.device)
    sin_out = torch.empty(output_shape, dtype=torch.float16, device=in_2.device)
    
    # Flatten input for kernel
    in_2_flat = in_2.view(-1)
    cos_out_flat = cos_out.view(-1)
    sin_out_flat = sin_out.view(-1)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_sin_cos_kernel[(num_programs,)](
        in_ptr=in_2_flat,
        cos_out_ptr=cos_out_flat,
        sin_out_ptr=sin_out_flat,
        seq_len=seq_len,
        orig_dim=orig_dim,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Stack outputs into single tensor [2, 1, seq_len, 32]
    stacked = torch.stack([cos_out, sin_out], dim=0)
    return stacked


def replacement_func():
    return fused_sin_cos_kernel_wrapper