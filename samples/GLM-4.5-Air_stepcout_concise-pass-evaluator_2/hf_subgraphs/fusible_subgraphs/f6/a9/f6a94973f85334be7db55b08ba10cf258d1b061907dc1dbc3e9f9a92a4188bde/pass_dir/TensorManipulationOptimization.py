import torch
import triton
import triton.language as tl

def pattern(conv_out, cls_token, pos_embed):
    # Original sequence:
    # tmp_9 = conv_out.flatten(2)  # [1, 768, 196] -> [1, 196, 768]
    # tmp_10 = tmp_9.transpose(1, 2)  
    # tmp_11 = cls_token.expand(1, -1, -1)
    # tmp_12 = torch.cat([tmp_11, tmp_10], dim=1)
    # tmp_13 = tmp_12 + pos_embed
    
    flattened = conv_out.flatten(2)
    transposed = flattened.transpose(1, 2)
    expanded = cls_token.expand(1, -1, -1)
    concatenated = torch.cat([expanded, transposed], dim=1)
    result = concatenated + pos_embed
    
    return concatenated, result

def replacement_args(conv_out, cls_token, pos_embed):
    # Get tensor shapes and dimensions
    conv_shape = conv_out.shape
    cls_shape = cls_token.shape
    pos_shape = pos_embed.shape
    
    hidden_size = conv_shape[1]  # 768 or 1408
    n_patches = conv_shape[2] * conv_shape[3]  # 14*14 = 196 or similar
    seq_len = pos_shape[1]  # 197 (197 = 1 cls + 196 patches) or 257
    
    return (conv_out, cls_token, pos_embed, hidden_size, n_patches, seq_len)

@triton.jit
def tensor_ops_kernel(
    conv_out_ptr,
    cls_token_ptr, pos_embed_ptr,
    out_cat_ptr, out_add_ptr,
    n_elements, hidden_size, n_patches, seq_len,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row in the final sequence
    row_idx = tl.program_id(0)
    
    # Determine which part of the sequence this row corresponds to
    if row_idx < 1:
        # CLS token row
        cls_offset = row_idx * hidden_size
        cls_data = tl.load(cls_token_ptr + cls_offset, mask=cls_offset < hidden_size).to(tl.float32)
        
        # Load pos_embed for CLS token
        pos_offset = row_idx * hidden_size
        pos_data = tl.load(pos_embed_ptr + pos_offset, mask=pos_offset < hidden_size).to(tl.float32)
        
        # Store to both outputs (cat result has CLS token, add result has sum)
        tl.store(out_cat_ptr + cls_offset, cls_data, mask=cls_offset < hidden_size)
        tl.store(out_add_ptr + cls_offset, cls_data + pos_data, mask=cls_offset < hidden_size)
    else:
        # Patch row (row_idx - 1 because CLS token is row 0)
        patch_idx = row_idx - 1
        
        # Load patch data from conv_out after flatten/transpose
        patch_offset_in_conv = patch_idx * hidden_size
        conv_patch_ptr = conv_out_ptr + patch_offset_in_conv
        patch_data = tl.load(conv_patch_ptr, mask=patch_offset_in_conv < n_patches * hidden_size).to(tl.float32)
        
        # Load pos_embed for this patch
        pos_offset = row_idx * hidden_size
        pos_data = tl.load(pos_embed_ptr + pos_offset, mask=pos_offset < seq_len * hidden_size).to(tl.float32)
        
        # Store results
        tl.store(out_cat_ptr + pos_offset, patch_data, mask=patch_offset_in_conv < n_patches * hidden_size)
        tl.store(out_add_ptr + pos_offset, patch_data + pos_data, mask=patch_offset_in_conv < n_patches * hidden_size)

@torch.fx.wrap
def tensor_ops_optimization(conv_out, cls_token, pos_embed, hidden_size, n_patches, seq_len):
    total_elements = seq_len * hidden_size
    
    # Use BLOCK_SIZE that works well for hidden_size
    if hidden_size >= 768:
        block_size = 256
    else:
        block_size = 128
        
    num_programs = (total_elements + block_size - 1) // block_size
    
    # Create output tensors
    concatenated = torch.empty((1, seq_len, hidden_size), dtype=conv_out.dtype, device=conv_out.device)
    result = torch.empty((1, seq_len, hidden_size), dtype=conv_out.dtype, device=conv_out.device)
    
    # Reshape for contiguous memory access
    conv_out_flat = conv_out.flatten(2).transpose(1, 2).contiguous()  # [1, n_patches, hidden_size]
    cls_contiguous = cls_token.contiguous()  # [1, 1, hidden_size]
    pos_contiguous = pos_embed.contiguous()  # [1, seq_len, hidden_size]
    
    tensor_ops_kernel[num_programs](
        conv_out_ptr=conv_out_flat,
        cls_token_ptr=cls_contiguous,
        pos_embed_ptr=pos_contiguous,
        out_cat_ptr=concatenated,
        out_add_ptr=result,
        n_elements=total_elements,
        hidden_size=hidden_size,
        n_patches=n_patches,
        seq_len=seq_len,
        BLOCK_SIZE=block_size
    )
    
    return concatenated, result

def replacement_func():
    return tensor_ops_optimization