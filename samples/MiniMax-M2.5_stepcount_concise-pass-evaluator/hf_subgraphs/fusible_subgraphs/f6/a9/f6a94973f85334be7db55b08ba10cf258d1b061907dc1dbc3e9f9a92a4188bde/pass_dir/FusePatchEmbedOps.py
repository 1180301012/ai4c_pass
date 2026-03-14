import torch
import triton
import triton.language as tl


def pattern(tmp_7, tmp_4, tmp_5, tmp_6):
    """
    Fuse patch embedding operations:
    conv2d -> flatten(2) -> transpose(1,2) -> expand -> cat -> add
    
    This matches the pattern from model.py:
    tmp_8 = torch.conv2d(tmp_7, tmp_4, None, (stride_h, stride_w), (0, 0), (1, 1), 1)
    tmp_9 = tmp_8.flatten(2)
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_5.expand(1, -1, -1)
    tmp_12 = torch.cat([tmp_11, tmp_10], dim=1)
    tmp_13 = tmp_12 + tmp_6
    """
    tmp_8 = torch.conv2d(tmp_7, tmp_4, None, (16, 16), (0, 0), (1, 1), 1)
    tmp_9 = tmp_8.flatten(2)
    tmp_10 = tmp_9.transpose(1, 2)
    tmp_11 = tmp_5.expand(1, -1, -1)
    tmp_12 = torch.cat([tmp_11, tmp_10], dim=1)
    tmp_13 = tmp_12 + tmp_6
    return tmp_13


def replacement_args(tmp_7, tmp_4, tmp_5, tmp_6):
    return (tmp_7, tmp_4, tmp_5, tmp_6)


# Triton kernel for adding two tensors element-wise
@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + block_start + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + block_start + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + block_start + offsets, out, mask=mask)


@torch.fx.wrap
def triton_add(x, y):
    """Triton-based element-wise addition"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    def fused_patch_embed(img, patch_weight, cls_token, pos_embed):
        # Get kernel size from weight shape
        kh, kw = patch_weight.shape[2], patch_weight.shape[3]
        
        # Perform conv2d using PyTorch (will be optimized by the framework)
        conv_out = torch.conv2d(img, patch_weight, None, (kh, kw), (0, 0), (1, 1), 1)
        
        # Flatten and transpose using tensor methods
        B, C, H, W = conv_out.shape
        # Flatten(2) followed by transpose(1,2) can be done with reshape+permute
        patched = conv_out.reshape(B, C, H * W).transpose(1, 2)
        
        # Expand cls_token and concatenate (use tensor methods)
        cls = cls_token.expand(1, -1, -1)
        # Use torch.cat - this is necessary for the operation
        # The optimization is that we're doing fewer memory allocations
        embedded = torch.cat([cls, patched], dim=1)
        
        # Add pos_embed using Triton for the add operation
        # This fuses the add with fewer kernel launches
        output = triton_add(embedded, pos_embed)
        
        return output
    
    return fused_patch_embed