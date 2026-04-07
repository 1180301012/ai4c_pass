import torch
import triton
import triton.language as tl

# Pattern matching function for Concat
def pattern(in_2, in_5, in_3):
    """
    Matches: torch.cat((in_2, in_5, in_3), dim=2)
    """
    result = torch.cat((in_2, in_5, in_3), dim=2)
    return result

# Extract arguments for replacement
def replacement_args(in_2, in_5, in_3):
    return (in_2, in_5, in_3)

# Optimized Concat kernel
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["hidden_size"],
)
@triton.jit
def concat_kernel(
    in_2_ptr,
    in_5_ptr,
    in_3_ptr,
    out_ptr,
    batch_size,
    cls_size,
    patch_size,
    det_size,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Concatenates three tensors along dim=2 (which is the third dimension).
    Tensor shapes:
      - in_2: [batch_size, 1, cls_size, hidden_size]
      - in_5: [batch_size, 1, patch_size, hidden_size]
      - in_3: [batch_size, 1, det_size, hidden_size]
      - out:  [batch_size, 1, (cls_size + patch_size + det_size), hidden_size]
    """
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Get pointer offsets for each input tensor (flattened view)
    ptr_2 = in_2_ptr + batch_idx * cls_size * hidden_size
    ptr_5 = in_5_ptr + batch_idx * patch_size * hidden_size
    ptr_3 = in_3_ptr + batch_idx * det_size * hidden_size
    out_ptr = out_ptr + batch_idx * (cls_size + patch_size + det_size) * hidden_size
    
    # Process in blocks
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_size
    
    # First segment: cls part
    if cls_size > 0:
        x = tl.load(ptr_2 + col_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + col_offsets, x, mask=mask)
    
    # Second segment: patch part
    if patch_size > 0:
        x = tl.load(ptr_5 + col_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + cls_size * hidden_size + col_offsets, x, mask=mask)
    
    # Third segment: det part
    if det_size > 0:
        x = tl.load(ptr_3 + col_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + (cls_size + patch_size) * hidden_size + col_offsets, x, mask=mask)

# Optimized wrapper for concat
@torch.fx.wrap
def concat_wrapper(in_2, in_5, in_3):
    """
    Optimized concat wrapper using Triton kernel.
    """
    # Get shapes
    batch_size = in_2.shape[0]
    cls_size = in_2.shape[2]
    det_size = in_3.shape[2]
    patch_size = in_5.shape[2]
    hidden_size = in_2.shape[3]
    
    # Compute output shape
    out_seq_len = cls_size + patch_size + det_size
    
    # Create output tensor
    out_shape = (batch_size, 1, out_seq_len, hidden_size)
    out = torch.empty(out_shape, device=in_2.device, dtype=in_2.dtype)
    
    # Flatten to 2D for the kernel
    in_2_flat = in_2.view(batch_size, -1)
    in_5_flat = in_5.view(batch_size, -1)
    in_3_flat = in_3.view(batch_size, -1)
    out_flat = out.view(batch_size, -1)
    
    # Determine BLOCK_SIZE based on hidden_size
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_size), 4096)
    
    # Launch kernel
    concat_kernel[(batch_size,)](
        in_2_ptr=in_2_flat,
        in_5_ptr=in_5_flat,
        in_3_ptr=in_3_flat,
        out_ptr=out_flat,
        batch_size=batch_size,
        cls_size=cls_size,
        patch_size=patch_size,
        det_size=det_size,
        hidden_size=hidden_size,
    )
    
    return out

def replacement_func():
    return concat_wrapper