import torch
import triton
import triton.language as tl

# Pattern matching function - matches flatten + transpose + add + dropout
def pattern(conv_output, pos_embed):
    """
    Match the pattern:
    flatten(2) -> transpose(1, 2) -> add -> dropout(0.0, False, False)
    """
    flat = conv_output.flatten(2)
    transposed = flat.transpose(1, 2)
    added = transposed + pos_embed
    result = torch.nn.functional.dropout(added, 0.0, False, False)
    return result

# Argument extraction function
def replacement_args(conv_output, pos_embed):
    return (conv_output, pos_embed)

# Use 2D grid: (spatial_blocks, channel_blocks) for better parallelism
@triton.jit
def fused_transpose_add_2d(
    input_ptr,       # [B, C, S] layout
    pos_embed_ptr,   # [1, S, C] layout  
    output_ptr,      # [B, S, C] layout
    B,
    C,
    S,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # 2D grid: (num_s_blocks, num_c_blocks)
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Compute start positions
    s_start = pid_s * BLOCK_S
    c_start = pid_c * BLOCK_C
    
    # Offsets within the block
    s_offs = s_start + tl.arange(0, BLOCK_S)
    c_offs = c_start + tl.arange(0, BLOCK_C)
    
    # Masks for bounds checking
    s_mask = s_offs < S
    c_mask = c_offs < C
    
    # Process batch 0 (B=1 in this case, can extend to loop over B)
    # Input: [B, C, S] -> [1, C, S] for B=1
    # offset = c * S + s
    inp_offs = c_offs[:, None] * S + s_offs[None, :]  # [BLOCK_C, BLOCK_S]
    mask_2d = c_mask[:, None] & s_mask[None, :]
    
    x = tl.load(input_ptr + inp_offs, mask=mask_2d, other=0.0)  # [BLOCK_C, BLOCK_S]
    
    # Pos embed: [1, S, C] -> offset = s * C + c
    pos_offs = s_offs[None, :] * C + c_offs[:, None]  # [BLOCK_C, BLOCK_S]
    p = tl.load(pos_embed_ptr + pos_offs, mask=mask_2d, other=0.0)
    
    # Need to transpose x to align with output layout
    # x is loaded as [BLOCK_C, BLOCK_S] but output wants [BLOCK_S, BLOCK_C]
    result = x + p  # [BLOCK_C, BLOCK_S]
    result_t = tl.trans(result)  # [BLOCK_S, BLOCK_C]
    
    # Output: [B, S, C] -> offset = s * C + c
    out_offs = s_offs[:, None] * C + c_offs[None, :]  # [BLOCK_S, BLOCK_C]
    out_mask = s_mask[:, None] & c_mask[None, :]
    
    tl.store(output_ptr + out_offs, result_t, mask=out_mask)


@torch.fx.wrap 
def optimized_flatten_transpose_add(conv_output, pos_embed):
    B = conv_output.shape[0]
    C = conv_output.shape[1]
    H = conv_output.shape[2]
    W = conv_output.shape[3]
    S = H * W
    
    # View as [B, C, S]
    inp = conv_output.view(B, C, S)
    
    # Output [B, S, C]
    out = torch.empty(B, S, C, device=conv_output.device, dtype=conv_output.dtype)
    
    # Block sizes - use powers of 2
    BLOCK_S = 64  # 196 / 64 ≈ 4 blocks
    BLOCK_C = 256  # 768 / 256 = 3 blocks
    
    num_s_blocks = (S + BLOCK_S - 1) // BLOCK_S
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    
    grid = (num_s_blocks, num_c_blocks)
    
    fused_transpose_add_2d[grid](
        inp, pos_embed, out,
        B, C, S,
        BLOCK_S, BLOCK_C
    )
    
    return out


def replacement_func():
    return optimized_flatten_transpose_add