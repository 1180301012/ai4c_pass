import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    # This matches the exact computation from the model
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(tmp_4)
    tmp_7 = tmp_6.float()
    tmp_8 = tmp_4 * tmp_7
    return (tmp_7, tmp_8, tmp_4)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def simple_mask_expansion_kernel(
    mask_ptr,          # attention mask [1, 16] 
    out_float_ptr,     # expanded mask output [1, 16, 768]
    n_tokens,          # 16
    hidden_size,       # 768
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one token in the sequence
    token_idx = tl.program_id(0)
    
    # Calculate offset for current token's hidden dimension
    token_offset = token_idx * hidden_size
    
    # Process hidden dimension in blocks
    hidden_offsets = token_offset + tl.arange(0, BLOCK_SIZE)
    mask = hidden_offsets < (n_tokens) * hidden_size
    
    # Load attention mask for this token
    mask_val = tl.load(mask_ptr + token_idx)
    
    # Create expanded attention mask (convert int64 to float32 and broadcast)
    expanded_mask = tl.full((BLOCK_SIZE,), float(mask_val), dtype=tl.float32)
    
    # Store results
    tl.store(out_float_ptr + hidden_offsets, expanded_mask, mask=mask)

@torch.fx.wrap
def simple_fused_computation(in_0, in_1, in_2, in_3):
    # Keep the original layer norm for now
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    
    # Optimize the mask expansion and type conversion with Triton
    n_tokens = in_3.size(1)  # 16
    hidden_size = in_3.size(2)  # 768
    
    # Output tensor for expanded mask
    out_float = torch.empty_like(in_3, dtype=torch.float32)
    
    # Choose block size for better GPU utilization
    BLOCK_SIZE = 256
    
    # Launch kernel for mask expansion
    simple_mask_expansion_kernel[(n_tokens,)](
        mask_ptr=in_0,
        out_float_ptr=out_float,
        n_tokens=n_tokens,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Perform multiplication
    tmp_8 = tmp_4 * out_float
    
    return (out_float, tmp_8, tmp_4)

def replacement_func():
    return simple_fused_computation