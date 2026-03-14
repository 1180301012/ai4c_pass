import torch
import triton
import triton.language as tl

def pattern(conv_out, cls_token):
    """
    Pattern: flatten + transpose + expand + cat
    conv_out: [batch, channels, h, w]
    cls_token: [1, 1, channels]
    
    After flatten(2): [batch, channels, h*w]
    After transpose(1, 2): [batch, h*w, channels]
    After expand: [batch, 1, channels]
    After cat: [batch, 1+h*w, channels]
    """
    flattened = conv_out.flatten(2)
    transposed = flattened.transpose(1, 2)
    expanded = cls_token.expand(1, -1, -1)
    result = torch.cat((expanded, transposed), dim=1)
    return result

def replacement_args(conv_out, cls_token):
    return (conv_out, cls_token)

@triton.jit
def fused_transpose_cat_kernel(
    conv_ptr,
    cls_ptr,
    output_ptr,
    batch,
    channels,
    spatial,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Ultra-optimized kernel: process entire channel dimension at once per spatial location
    """
    pid = tl.program_id(0)
    
    # Each program handles multiple elements
    total_output_elements = batch * (1 + spatial) * channels
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_output_elements
    
    # Decode output index: [batch, seq, channel]
    b_idx = idx // ((1 + spatial) * channels)
    remainder = idx % ((1 + spatial) * channels)
    seq_idx = remainder // channels
    c_idx = remainder % channels
    
    # If seq_idx == 0, load from cls_token; otherwise from conv
    is_cls = seq_idx == 0
    
    # For cls_token: just read cls_ptr[c_idx]
    cls_val = tl.load(cls_ptr + c_idx, mask=mask & is_cls, other=0.0)
    
    # For conv: read conv_ptr[b_idx, c_idx, seq_idx-1]
    spatial_idx = seq_idx - 1
    conv_offset = b_idx * (channels * spatial) + c_idx * spatial + spatial_idx
    conv_val = tl.load(conv_ptr + conv_offset, mask=mask & (~is_cls), other=0.0)
    
    # Select correct value
    output_val = tl.where(is_cls, cls_val, conv_val)
    
    # Store
    tl.store(output_ptr + idx, output_val, mask=mask)

@torch.fx.wrap
def fused_expand_flatten_transpose_cat(conv_out, cls_token):
    """
    Fused implementation with optimized Triton kernel
    """
    batch, channels, h, w = conv_out.shape
    spatial = h * w
    seq_len = 1 + spatial
    
    # Flatten conv_out to [batch, channels, spatial]
    conv_flat = conv_out.reshape(batch, channels, spatial)
    
    # Allocate output [batch, seq_len, channels]
    output = torch.empty((batch, seq_len, channels), dtype=conv_out.dtype, device=conv_out.device)
    
    # Launch kernel with large block size to minimize overhead
    total_elements = batch * seq_len * channels
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    fused_transpose_cat_kernel[grid](
        conv_flat,
        cls_token,
        output,
        batch,
        channels,
        spatial,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_expand_flatten_transpose_cat